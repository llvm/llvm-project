//===- SemaSYCL.cpp - Semantic Analysis for SYCL constructs ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for SYCL constructs.
//===----------------------------------------------------------------------===//

#include "TreeTransform.h"
#include "clang/AST/AST.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

typedef llvm::DenseMap<DeclaratorDecl *, DeclaratorDecl *> DeclMap;

using KernelParamKind = SYCLIntegrationHeader::kernel_param_kind_t;

enum target {
  global_buffer = 2014,
  constant_buffer,
  local,
  image,
  host_buffer,
  host_image,
  image_array
};

static CXXRecordDecl *getKernelCallerLambdaArg(FunctionDecl *FD) {
  auto FirstArg = (*FD->param_begin());
  if (FirstArg)
    if (FirstArg->getType()->getAsCXXRecordDecl()->isLambda())
      return FirstArg->getType()->getAsCXXRecordDecl();
  return nullptr;
}

class MarkDeviceFunction : public RecursiveASTVisitor<MarkDeviceFunction> {
public:
  MarkDeviceFunction(Sema &S)
      : RecursiveASTVisitor<MarkDeviceFunction>(), SemaRef(S) {}
  bool VisitCallExpr(CallExpr *e) {
    if (FunctionDecl *Callee = e->getDirectCallee()) {
      // Remember that all SYCL kernel functions have deferred
      // instantiation as template functions. It means that
      // all functions used by kernel have already been parsed and have
      // definitions.
      if (FunctionDecl *Def = Callee->getDefinition()) {
        if (!Def->hasAttr<SYCLDeviceAttr>()) {
          Def->addAttr(SYCLDeviceAttr::CreateImplicit(SemaRef.Context));
          this->TraverseStmt(Def->getBody());
          // But because parser works with top level declarations and CodeGen
          // already saw and ignored our function without device attribute we
          // need to add this function into SYCL kernels array to show it
          // this function again.
          SemaRef.AddSyclKernel(Def);
        }
      }
    }
    return true;
  }

private:
  Sema &SemaRef;
};

class KernelBodyTransform : public TreeTransform<KernelBodyTransform> {
public:
  KernelBodyTransform(llvm::DenseMap<DeclaratorDecl *, DeclaratorDecl *> &Map,
                      Sema &S)
      : TreeTransform<KernelBodyTransform>(S), DMap(Map), SemaRef(S) {}
  bool AlwaysRebuild() { return true; }

  ExprResult TransformDeclRefExpr(DeclRefExpr *DRE) {
    auto Ref = dyn_cast<DeclaratorDecl>(DRE->getDecl());
    if (Ref) {
      auto NewDecl = DMap[Ref];
      if (NewDecl) {
        return DeclRefExpr::Create(
            SemaRef.getASTContext(), DRE->getQualifierLoc(),
            DRE->getTemplateKeywordLoc(), NewDecl, false, DRE->getNameInfo(),
            NewDecl->getType(), DRE->getValueKind());
      }
    }
    return DRE;
  }

private:
  DeclMap DMap;
  Sema &SemaRef;
};

static FunctionDecl *
CreateSYCLKernelFunction(ASTContext &Context, StringRef Name,
                         ArrayRef<QualType> ArgTys,
                         ArrayRef<DeclaratorDecl *> ArgDecls) {

  DeclContext *DC = Context.getTranslationUnitDecl();
  FunctionProtoType::ExtProtoInfo Info(CC_OpenCLKernel);
  QualType RetTy = Context.VoidTy;
  QualType FuncTy = Context.getFunctionType(RetTy, ArgTys, Info);
  DeclarationName DN = DeclarationName(&Context.Idents.get(Name));
  FunctionDecl *SYCLKernel = FunctionDecl::Create(
      Context, DC, SourceLocation(), SourceLocation(), DN, FuncTy,
      Context.getTrivialTypeSourceInfo(RetTy), SC_None);
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  int i = 0;
  for (auto ArgTy : ArgTys) {
    auto P = ParmVarDecl::Create(Context, SYCLKernel, SourceLocation(),
                                 SourceLocation(), ArgDecls[i]->getIdentifier(),
                                 ArgTy, ArgDecls[i]->getTypeSourceInfo(),
                                 SC_None, 0);
    P->setScopeInfo(0, i++);
    P->setIsUsed();
    Params.push_back(P);
  }
  SYCLKernel->setParams(Params);

  SYCLKernel->addAttr(SYCLDeviceAttr::CreateImplicit(Context));
  SYCLKernel->addAttr(OpenCLKernelAttr::CreateImplicit(Context));
  SYCLKernel->addAttr(AsmLabelAttr::CreateImplicit(Context, Name));
  // To see kernel in AST-dump.
  DC->addDecl(SYCLKernel);
  return SYCLKernel;
}

static CompoundStmt *
CreateSYCLKernelBody(Sema &S, FunctionDecl *KernelCallerFunc, DeclContext *DC) {

  llvm::SmallVector<Stmt *, 16> BodyStmts;

  // TODO: case when kernel is functor
  // TODO: possible refactoring when functor case will be completed
  CXXRecordDecl *LC = getKernelCallerLambdaArg(KernelCallerFunc);
  if (LC) {
    // Create Lambda object
    auto LambdaVD = VarDecl::Create(
        S.Context, DC, SourceLocation(), SourceLocation(), LC->getIdentifier(),
        QualType(LC->getTypeForDecl(), 0), LC->getLambdaTypeInfo(), SC_None);

    Stmt *DS = new (S.Context)
        DeclStmt(DeclGroupRef(LambdaVD), SourceLocation(), SourceLocation());
    BodyStmts.push_back(DS);
    auto LambdaDRE = DeclRefExpr::Create(
        S.Context, NestedNameSpecifierLoc(), SourceLocation(), LambdaVD, false,
        DeclarationNameInfo(), QualType(LC->getTypeForDecl(), 0), VK_LValue);

    // Initialize Lambda fields
    llvm::SmallVector<Expr *, 16> InitCaptures;

    auto TargetFunc = dyn_cast<FunctionDecl>(DC);
    auto TargetFuncParam =
        TargetFunc->param_begin(); // Iterator to ParamVarDecl (VarDecl)
    if (TargetFuncParam) {
      for (auto Field : LC->fields()) {
        QualType ParamType = (*TargetFuncParam)->getOriginalType();
        auto DRE =
            DeclRefExpr::Create(S.Context, NestedNameSpecifierLoc(),
                                SourceLocation(), *TargetFuncParam, false,
                                DeclarationNameInfo(), ParamType, VK_LValue);

        QualType FieldType = Field->getType();
        CXXRecordDecl *CRD = FieldType->getAsCXXRecordDecl();
        if (CRD) {
          llvm::SmallVector<Expr *, 16> ParamStmts;
          DeclAccessPair FieldDAP = DeclAccessPair::make(Field, AS_none);
          auto AccessorME = MemberExpr::Create(
              S.Context, LambdaDRE, false, SourceLocation(),
              NestedNameSpecifierLoc(), SourceLocation(), Field, FieldDAP,
              DeclarationNameInfo(Field->getDeclName(), SourceLocation()),
              nullptr, Field->getType(), VK_LValue, OK_Ordinary);

          for (auto Method : CRD->methods()) {
            if (Method->getNameInfo().getName().getAsString() ==
                "__set_pointer") {
              DeclAccessPair MethodDAP = DeclAccessPair::make(Method, AS_none);
              auto ME = MemberExpr::Create(
                  S.Context, AccessorME, false, SourceLocation(),
                  NestedNameSpecifierLoc(), SourceLocation(), Method, MethodDAP,
                  Method->getNameInfo(), nullptr, Method->getType(), VK_LValue,
                  OK_Ordinary);

              // Not referenced -> not emitted
              S.MarkFunctionReferenced(SourceLocation(), Method, true);

              QualType ResultTy = Method->getReturnType();
              ExprValueKind VK = Expr::getValueKindForType(ResultTy);
              ResultTy = ResultTy.getNonLValueExprType(S.Context);

              // __set_pointer needs one parameter
              QualType paramTy = (*(Method->param_begin()))->getOriginalType();

              // C++ address space attribute != OpenCL address space attribute
              Expr *qualifiersCast = ImplicitCastExpr::Create(
                  S.Context, paramTy, CK_NoOp, DRE, nullptr, VK_LValue);
              Expr *Res = ImplicitCastExpr::Create(
                  S.Context, paramTy, CK_LValueToRValue, qualifiersCast,
                  nullptr, VK_RValue);

              ParamStmts.push_back(Res);

              // lambda.accessor.__set_pointer(kernel_parameter)
              CXXMemberCallExpr *Call = CXXMemberCallExpr::Create(
                  S.Context, ME, ParamStmts, ResultTy, VK, SourceLocation());
              BodyStmts.push_back(Call);
            }
          }
        } else if (FieldType->isBuiltinType()) {
          // If field have built-in type just initialize this field
          // with corresponding kernel argument using '=' binary operator.
          DeclAccessPair FieldDAP = DeclAccessPair::make(Field, AS_none);
          auto Lhs = MemberExpr::Create(
              S.Context, LambdaDRE, false, SourceLocation(),
              NestedNameSpecifierLoc(), SourceLocation(), Field, FieldDAP,
              DeclarationNameInfo(Field->getDeclName(), SourceLocation()),
              nullptr, Field->getType(), VK_LValue, OK_Ordinary);
          auto Rhs = ImplicitCastExpr::Create(
              S.Context, ParamType, CK_LValueToRValue, DRE, nullptr, VK_RValue);
          // lambda.field = kernel_parameter
          Expr *Res = new (S.Context)
              BinaryOperator(Lhs, Rhs, BO_Assign, FieldType, VK_LValue,
                             OK_Ordinary, SourceLocation(), FPOptions());
          BodyStmts.push_back(Res);
        }
        TargetFuncParam++;
      }
    }

    // In function from headers lambda is function parameter, we need
    // to replace all refs to this lambda with our vardecl.
    // I used TreeTransform here, but I'm not sure that it is good solution
    // Also I used map and I'm not sure about it too.
    // TODO SYCL review the above design concerns
    Stmt *FunctionBody = KernelCallerFunc->getBody();
    DeclMap DMap;
    ParmVarDecl *LambdaParam = *(KernelCallerFunc->param_begin());
    // DeclRefExpr with valid source location but with decl which is not marked
    // as used is invalid.
    LambdaVD->setIsUsed();
    DMap[LambdaParam] = LambdaVD;
    // Without PushFunctionScope I had segfault. Maybe we also need to do pop.
    S.PushFunctionScope();
    KernelBodyTransform KBT(DMap, S);
    Stmt *NewBody = KBT.TransformStmt(FunctionBody).get();
    BodyStmts.push_back(NewBody);
  }
  return CompoundStmt::Create(S.Context, BodyStmts, SourceLocation(),
                              SourceLocation());
}

/// Various utilities.
class Util {
public:
  // TODO SYCL use AST infrastructure instead of string matching

  /// Checks whether given clang type is a sycl accessor class.
  static bool isSyclAccessorType(QualType Ty) {
    std::string Name = Ty.getCanonicalType().getAsString();
    return Name.find("class cl::sycl::accessor") != std::string::npos;
  }

  /// Checks whether given clang type is a sycl stream class.
  static bool isSyclStreamType(QualType Ty) {
    std::string Name = Ty.getCanonicalType().getAsString();
    return Name == "stream";
  }
};

/// Identifies context of kernel lambda capture visitor function
/// invocation.
enum VisitorContext {
  pre_visit,
  visit_accessor,
  visit_scalar,
  visit_stream,
  post_visit,
};

/// Implements visitor design pattern for lambda captures.
///
/// Iterates over captured parameters of given lambda and invokes given
/// visitor functions at appropriate context providing information of interest.
/// \param Lambda  the kernel lambda object
/// \param Vis     a tuple of visitor functions, each corresponds to and is
///     invoked at a specific context. @see VisitorContext.
///
template <typename VisitorTupleTy>
static void visitKernelLambdaCaptures(const CXXRecordDecl *Lambda,
                                      VisitorTupleTy &Vis) {
  const LambdaCapture *Cpt = Lambda->captures_begin();
  RecordDecl::field_iterator Fld = Lambda->field_begin();
  const LambdaCapture *CptEnd = Lambda->captures_end();
  const RecordDecl::field_iterator FldEnd = Lambda->field_end();

  for (; (Cpt != CptEnd) && (Fld != FldEnd); Cpt++, Fld++) {
    // pre-visit context
    unsigned Cnt = static_cast<unsigned>(std::distance(Cpt, CptEnd));
    VarDecl *V = Cpt->getCapturedVar();
    QualType ArgTy = V->getType();
    auto F1 = std::get<pre_visit>(Vis);
    F1(Cnt, V, *Fld);

    if (Util::isSyclAccessorType(ArgTy)) {
      // accessor parameter context
      const auto *RecordDecl = ArgTy->getAsCXXRecordDecl();
      assert(RecordDecl && "accessor must be of a record type");
      const auto *TemplateDecl =
          dyn_cast<ClassTemplateSpecializationDecl>(RecordDecl);
      assert(TemplateDecl && "templated accessor type expected");

      // First accessor template parameter - data type
      QualType PointeeType = TemplateDecl->getTemplateArgs()[0].getAsType();
      // Fourth parameter - access target
      auto AccessQualifier = TemplateDecl->getTemplateArgs()[3].getAsIntegral();
      int64_t AccessTarget = AccessQualifier.getExtValue();
      auto F = std::get<visit_accessor>(Vis);
      F(Cnt, static_cast<target>(AccessTarget), PointeeType, V, *Fld);
    } else if (Util::isSyclStreamType(ArgTy)) {
      // stream parameter context
      auto F = std::get<visit_stream>(Vis);
      F(Cnt, V, *Fld);
    } else if (ArgTy->isScalarType()) {
      // scalar typed parameter context
      auto F = std::get<visit_scalar>(Vis);
      F(Cnt, V, *Fld);
    } else {
      llvm_unreachable("unsupported kernel parameter type");
    }
    // pos-visit context
    auto F2 = std::get<post_visit>(Vis);
    F2(Cnt, V, *Fld);
  }
  assert((Cpt == CptEnd) && (Fld == FldEnd) &&
         "captures inconsistent with fields");
}

static void BuildArgTys(ASTContext &Context, CXXRecordDecl *Lambda,
                        llvm::SmallVector<DeclaratorDecl *, 16> &NewArgDecls,
                        llvm::SmallVector<QualType, 16> &ArgTys) {
  QualType ActualArgType; // serves to transfer info between visitor lambdas
  auto Vis = std::make_tuple(
      // pre_visit
      [&](int, VarDecl *, FieldDecl *) {},
      // visit_accessor
      [&](int CaptureN, target AccTrg, QualType PointeeType,
          DeclaratorDecl *CapturedVar, FieldDecl *CapturedVal) {
        Qualifiers Quals = PointeeType.getQualifiers();
        // TODO: Support all access targets
        switch (AccTrg) {
        case target::global_buffer:
          Quals.setAddressSpace(LangAS::opencl_global);
          break;
        case target::constant_buffer:
          Quals.setAddressSpace(LangAS::opencl_constant);
          break;
        case target::local:
          Quals.setAddressSpace(LangAS::opencl_local);
          break;
        default:
          llvm_unreachable("Unsupported access target");
        }
        // TODO: get address space from accessor template parameter.
        PointeeType =
            Context.getQualifiedType(PointeeType.getUnqualifiedType(), Quals);
        QualType PointerType = Context.getPointerType(PointeeType);
        ActualArgType =
            Context.getQualifiedType(PointerType.getUnqualifiedType(), Quals);
      },
      // visit_scalar
      [&](int CaptureN, VarDecl *CapturedVar, FieldDecl *CapturedVal) {
        ActualArgType = CapturedVal->getType();
      },
      // visit_stream
      [&](int CaptureN, VarDecl *CapturedVar, FieldDecl *CapturedVal) {
        llvm_unreachable("streams not supported yet");
      },
      // post_visit
      [&](int CaptureN, VarDecl *CapturedVar, FieldDecl *CapturedVal) {
        DeclContext *DC = Context.getTranslationUnitDecl();

        IdentifierInfo *VarName = 0;
        SmallString<8> Str;
        llvm::raw_svector_ostream OS(Str);
        OS << "_arg_" << CapturedVar->getIdentifier()->getName();
        VarName = &Context.Idents.get(OS.str());

        auto NewVarDecl = VarDecl::Create(
            Context, DC, SourceLocation(), SourceLocation(), VarName,
            ActualArgType, Context.getTrivialTypeSourceInfo(ActualArgType),
            SC_None);
        ArgTys.push_back(ActualArgType);
        NewArgDecls.push_back(NewVarDecl);
      });
  visitKernelLambdaCaptures(Lambda, Vis);
}

/// Adds necessary data describing given kernel to the integration header.
/// \param H        the integration header object
/// \param Name     kernel name
/// \param NameType user-specified type representing kernel name
/// \param Lambda   kernel lambda object
static void populateIntHeader(SYCLIntegrationHeader &H, const StringRef Name,
                              QualType NameType, CXXRecordDecl *Lambda) {
  ASTContext &Ctx = Lambda->getASTContext();
  const ASTRecordLayout &Layout = Ctx.getASTRecordLayout(Lambda);
  KernelParamKind Knd = SYCLIntegrationHeader::kind_last;
  H.startKernel(Name, NameType);
  unsigned Offset = 0;
  int Info = 0;

  auto Vis = std::make_tuple(
      // pre_visit
      [&](int CaptureN, VarDecl *CapturedVar, FieldDecl *CapturedVal) {
        // Set offset in bytes
        Offset = static_cast<unsigned>(
            Layout.getFieldOffset(CapturedVal->getFieldIndex()))/8;
      },
      // visit_accessor
      [&](int CaptureN, target AccTrg, QualType PointeeType,
          DeclaratorDecl *CapturedVar, FieldDecl *CapturedVal) {
        Knd = SYCLIntegrationHeader::kind_accessor;
        Info = static_cast<int>(AccTrg);
      },
      // visit_scalar
      [&](int CaptureN, VarDecl *CapturedVar, FieldDecl *CapturedVal) {
        Knd = SYCLIntegrationHeader::kind_scalar;
        Info = static_cast<unsigned>(
            Ctx.getTypeSizeInChars(CapturedVal->getType()).getQuantity());
      },
      // visit_stream
      [&](int CaptureN, VarDecl *CapturedVar, FieldDecl *CapturedVal) {
        llvm_unreachable("streams not supported yet");
      },
      // post_visit
      [&](int CaptureN, VarDecl *CapturedVar, FieldDecl *CapturedVal) {
        H.addParamDesc(Knd, Info, Offset);
      });
  visitKernelLambdaCaptures(Lambda, Vis);
}

// Creates a kernel name for given kernel name type which is unique across all
// instantiations of the type if it is templated. If it is not templated,
// uniqueueness is prescribed by the SYCL spec. 'class' and 'struct' keywords
// are removed to make the name shorter. Non-alphanumeric characters in a kernel
// name are OK - SPIRV and runtimes allow that.
static std::string constructKernelName(QualType KernelNameType) {
  static const std::string Kwds[] = {
    std::string("class"),
    std::string("struct")
  };
  std::string TStr = KernelNameType.getAsString();

  for (const std::string &Kwd : Kwds) {
    for (size_t Pos = TStr.find(Kwd);
         Pos != StringRef::npos;
         Pos = TStr.find(Kwd, Pos)) {

      size_t EndPos = Pos + Kwd.length();
      if ((!llvm::isAlnum(TStr[Pos - 1])) &&
          (EndPos == TStr.length() || !llvm::isAlnum(TStr[EndPos]))) {
        // keyword is a separate word - erase
        TStr.erase(Pos, Kwd.length());
      } else
        Pos = EndPos;
    }
  }
  return StringRef(TStr).trim().str();
}

void Sema::ConstructSYCLKernel(FunctionDecl *KernelCallerFunc) {
  // TODO: Case when kernel is functor
  CXXRecordDecl *LE = getKernelCallerLambdaArg(KernelCallerFunc);
  if (LE) {
    llvm::SmallVector<QualType, 16> ArgTys;
    llvm::SmallVector<DeclaratorDecl *, 16> NewArgDecls;
    BuildArgTys(getASTContext(), LE, NewArgDecls, ArgTys);

    // Get Name for our kernel.
    const TemplateArgumentList *TemplateArgs =
        KernelCallerFunc->getTemplateSpecializationArgs();
    QualType KernelNameType = TemplateArgs->get(0).getAsType();
    std::string Name = constructKernelName(KernelNameType);
    populateIntHeader(getSyclIntegrationHeader(), Name, KernelNameType, LE);

    FunctionDecl *SYCLKernel =
        CreateSYCLKernelFunction(getASTContext(), Name, ArgTys, NewArgDecls);

    CompoundStmt *SYCLKernelBody =
        CreateSYCLKernelBody(*this, KernelCallerFunc, SYCLKernel);
    SYCLKernel->setBody(SYCLKernelBody);

    AddSyclKernel(SYCLKernel);

    // Let's mark all called functions with SYCL Device attribute.
    MarkDeviceFunction Marker(*this);
    Marker.TraverseStmt(SYCLKernelBody);
  }
}

// -----------------------------------------------------------------------------
// Integration header functionality implementation
// -----------------------------------------------------------------------------

/// Returns a string ID of given parameter kind - used in header
/// emission.
static const char *paramKind2Str(KernelParamKind K) {
#define CASE(x)                                                                \
  case SYCLIntegrationHeader::kind_##x:                                        \
    return "kind_" #x
  switch (K) {
    CASE(accessor);
    CASE(scalar);
    CASE(struct);
    CASE(sampler);
    CASE(struct_padding);
  default:
    return "<ERROR>";
  }
#undef CASE
}

// Prints a declaration
static void printDecl(raw_ostream &O, const Decl *D) {
  PrintingPolicy P(D->getASTContext().getLangOpts());
  // print declaration into a string:
  P.TerseOutput = true; // prints declaration plus " {}" in the end
  P.PolishForDeclaration = true;
  std::string S;
  llvm::raw_string_ostream SO(S);
  D->print(SO, P);
  // print the declaration w/o the trailing " {}":
  StringRef SR = SO.str();
  size_t Pos = SR.find_first_of('{');

  if (Pos != StringRef::npos) {
    // can be npos if the type is incomplete
    SR = SR.take_front(Pos);
  }
  O << SR;
}

// Emits forward declarations of classes and template classes on which
// declaration of given type depends. For example, consider SimpleVadd
// class specialization in parallel_for below:
//
//   template <typename T1, unsigned int N, typename ... T2>
//   class SimpleVadd;
//   ...
//   template <unsigned int N, typename T1, typename ... T2>
//   void simple_vadd(const std::array<T1, N>& VA, const std::array<T1, N>& VB,
//     std::array<T1, N>& VC, int param, T2 ... varargs) {
//     ...
//     deviceQueue.submit([&](cl::sycl::handler& cgh) {
//       ...
//       cgh.parallel_for<class SimpleVadd<T1, N, T2...>>(...)
//       ...
//     }
//     ...
//   }
//   ...
//   class MyClass {...};
//   template <typename T> class MyInnerTmplClass { ... }
//   template <typename T> class MyTmplClass { ... }
//   ...
//   MyClass *c = new MyClass();
//   MyInnerTmplClass<MyClass**> c1(&c);
//   simple_vadd(A, B, C, 5, 'a', 1.f,
//     new MyTmplClass<MyInnerTmplClass<MyClass**>>(c1));
//
// it will generate the following forward declarations:
//   class MyClass;
//   template <typename T> class MyInnerTmplClass;
//   template <typename T> class MyTmplClass;
//   template <typename T1, unsigned int N, typename ...T2> class SimpleVadd;
//
// TODO FIXME handle the case when kernel typename is declared in a namespace
//
// \param O
//     stream to print to
// \param T
//     type to emit forward declarations for
// \param Printed
//     a set of type pointers forward declrations has been printed for already
// \param Depth
//     recursion depth
//
static void emitForwardClassDecls(raw_ostream &O,
                                  QualType T,
                                  llvm::SmallPtrSetImpl<const void*> &Printed) {

  // peel off the pointer types and get the class/struct type:
  for (; T->isPointerType(); T = T->getPointeeType());
  const CXXRecordDecl *RD = T->getAsCXXRecordDecl();

  if (!RD)
    return;

  // see if this is a template specialization ...
  if (const auto *TSD = dyn_cast<ClassTemplateSpecializationDecl>(RD)) {
    // ... yes, it is template specialization:
    // - first, recurse into template parameters and emit needed forward
    //   declarations
    const TemplateArgumentList &Args = TSD->getTemplateArgs();

    for (unsigned I = 0; I < Args.size(); I++) {
      const TemplateArgument &Arg = Args[I];

      switch (Arg.getKind()) {
      case TemplateArgument::ArgKind::Type:
        emitForwardClassDecls(O, Arg.getAsType(), Printed);
        break;
      case TemplateArgument::ArgKind::Pack: {
        ArrayRef<TemplateArgument> Pack = Arg.getPackAsArray();

        for (const auto &T : Pack) {
          if (T.getKind() == TemplateArgument::ArgKind::Type) {
            emitForwardClassDecls(O, T.getAsType(), Printed);
          }
        }
        break;
      }
      case TemplateArgument::ArgKind::Template:
        llvm_unreachable("template template arguments not supported");
      default:
        break; // nop
      }
    }
    // - second, emit forward declaration for the template class being
    //   specialized
    ClassTemplateDecl *CTD = TSD->getSpecializedTemplate();
    assert(CTD && "template declaration must be available");

    if (Printed.insert(CTD).second) {
      printDecl(O, CTD);
      O << ";\n";
    }
  } else if (Printed.insert(RD).second) {
    // emit forward declarations for "leaf" classes in the template parameter
    // tree; Depth > 0: don't print forward decl for top level non-templated
    // class
    printDecl(O, RD);
    O << ";\n";
  }
}

void SYCLIntegrationHeader::emit(raw_ostream &O) {
  O << "// This is auto-generated SYCL integration header.\n";
  O << "\n";
  O << "// Forward declarations of templated kernel function types:\n";

  llvm::SmallPtrSet<const void*, 4> Printed;

  for (const KernelDesc &K : KernelDescs) {
    emitForwardClassDecls(O, K.NameType, Printed);
  }
  O << "\n";

  O << "#include <CL/sycl/detail/kernel_desc.hpp>\n";

  O << "\n";

  O << "namespace cl {\n";
  O << "namespace sycl {\n";
  O << "namespace detail {\n";

  O << "\n";

  O << "// names of all kernels defined in the corresponding source\n";
  O << "static constexpr\n";
  O << "const char* const kernel_names[] = {\n";

  for (unsigned I = 0; I < KernelDescs.size(); I++) {
    O << "  \"" << KernelDescs[I].Name << "\"";

    if (I < KernelDescs.size() - 1)
      O << ",";
    O << "\n";
  }
  O << "};\n\n";

  O << "// array representing signatures of all kernels defined in the\n";
  O << "// corresponding source\n";
  O << "static constexpr\n";
  O << "const kernel_param_desc_t kernel_signatures[] = {\n";

  for (unsigned I = 0; I < KernelDescs.size(); I++) {
    auto &K = KernelDescs[I];
    O << "  //--- " << K.Name << "\n";

    for (const auto &P : K.Params) {
      std::string TyStr = paramKind2Str(P.Kind);
      O << "  { kernel_param_kind_t::" << TyStr << ", ";
      O << P.Info << ", " << P.Offset << " },\n";
    }
    O << "\n";
  }
  O << "};\n\n";

  O << "// indices into the kernel_signatures array, each representing a start"
       " of\n";
  O << "// kernel signature descriptor subarray of the kernel_signatures"
       " array;\n";
  O << "// the index order in this array corresponds to the kernel name order"
       " in the\n";
  O << "// kernel_names array\n";
  O << "static constexpr\n";
  O << "const unsigned kernel_signature_start[] = {\n";
  unsigned CurStart = 0;

  for (unsigned I = 0; I < KernelDescs.size(); I++) {
    auto &K = KernelDescs[I];
    O << "  " << CurStart;
    if (I < KernelDescs.size() - 1)
      O << ",";
    O << " // " << K.Name << "\n";
    CurStart += K.Params.size() + 1;
  }
  O << "};\n\n";

  O << "// Specializations of this template class encompasses information\n";
  O << "// about a kernel. The kernel is identified by the template\n";
  O << "// parameter type.\n";
  O << "template <class KernelNameType> struct KernelInfo;\n";
  O << "\n";

  O << "// Specializations of KernelInfo for kernel function types:\n";
  CurStart = 0;

  for (const KernelDesc &K : KernelDescs) {
    const size_t N = K.Params.size();
    O << "template <> struct KernelInfo<" <<
      K.NameType.getAsString() << "> {\n";
    O << "  static constexpr const char* getName() { return \""
      << K.Name << "\"; }\n";
    O << "  static constexpr unsigned getNumParams() { return "
      << N << "; }\n";
    O << "  static constexpr const kernel_param_desc_t& ";
    O << "getParamDesc(unsigned i) {\n";
    O << "    return kernel_signatures[i+" << CurStart << "];\n";
    O << "  }\n";
    O << "};\n";
    CurStart += N;
  }
  O << "\n";
  O << "} // namespace detail\n";
  O << "} // namespace sycl\n";
  O << "} // namespace cl\n";
  O << "\n";
}


bool SYCLIntegrationHeader::emit(const StringRef &IntHeaderName) {
  if (IntHeaderName.empty())
    return false;
  int IntHeaderFD = 0;
  std::error_code EC =
      llvm::sys::fs::openFileForWrite(IntHeaderName, IntHeaderFD);
  if (EC) {
    llvm::errs() << "Error: " << EC.message() << "\n";
    // compilation will fail on absent include file - don't need to fail here
    return false;
  }
  llvm::raw_fd_ostream Out(IntHeaderFD, true /*close in destructor*/);
  emit(Out);
  return true;
}

void SYCLIntegrationHeader::startKernel(StringRef KernelName,
                                        QualType KernelNameType) {
  KernelDescs.resize(KernelDescs.size() + 1);
  KernelDescs.back().Name = KernelName;
  KernelDescs.back().NameType = KernelNameType;
}

void SYCLIntegrationHeader::addParamDesc(kernel_param_kind_t Kind, int Info,
                                         unsigned Offset) {
  auto *K = getCurKernelDesc();
  assert(K && "no kernels");
  K->Params.push_back(KernelParamDesc());
  KernelParamDesc &PD = K->Params.back();
  PD.Kind = Kind;
  PD.Info = Info;
  PD.Offset = Offset;
}

void SYCLIntegrationHeader::endKernel() {
  // nop for now
}

SYCLIntegrationHeader::SYCLIntegrationHeader() {}
