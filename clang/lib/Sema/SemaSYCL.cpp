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

#include "clang/AST/AST.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallVector.h"
#include "TreeTransform.h"

using namespace clang;

typedef llvm::DenseMap<DeclaratorDecl *, DeclaratorDecl *> DeclMap;

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

CXXRecordDecl* getBodyAsLambda(FunctionDecl *FD) {
  auto FirstArg = (*FD->param_begin());
  if (FirstArg)
    if (FirstArg->getType()->getAsCXXRecordDecl()->isLambda())
      return FirstArg->getType()->getAsCXXRecordDecl();
  return nullptr;
}

FunctionDecl *CreateSYCLKernelFunction(ASTContext &Context, StringRef Name,
                                       ArrayRef<QualType> ArgTys,
                                       ArrayRef<DeclaratorDecl *> ArgDecls) {

  DeclContext *DC = Context.getTranslationUnitDecl();
  FunctionProtoType::ExtProtoInfo Info(CC_OpenCLKernel);
  QualType RetTy = Context.VoidTy;
  QualType FuncTy = Context.getFunctionType(RetTy, ArgTys, Info);
  DeclarationName DN = DeclarationName(&Context.Idents.get(Name));
  FunctionDecl *Result = FunctionDecl::Create(
      Context, DC, SourceLocation(), SourceLocation(), DN, FuncTy,
      Context.getTrivialTypeSourceInfo(RetTy), SC_None);
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  int i = 0;
  for (auto ArgTy : ArgTys) {
    auto P =
        ParmVarDecl::Create(Context, Result, SourceLocation(), SourceLocation(),
                            ArgDecls[i]->getIdentifier(), ArgTy,
                            ArgDecls[i]->getTypeSourceInfo(), SC_None, 0);
    P->setScopeInfo(0, i++);
    P->setIsUsed();
    Params.push_back(P);
  }
  Result->setParams(Params);
  // TODO: Add SYCL specific attribute for kernel and all functions called
  // by kernel.
  Result->addAttr(SYCLDeviceAttr::CreateImplicit(Context));
  Result->addAttr(OpenCLKernelAttr::CreateImplicit(Context));
  Result->addAttr(AsmLabelAttr::CreateImplicit(Context, Name));
  // To see kernel in ast-dump.
  DC->addDecl(Result);
  return Result;
}

CompoundStmt *CreateSYCLKernelBody(Sema &S, FunctionDecl *KernelHelper,
                                   DeclContext *DC) {

  llvm::SmallVector<Stmt *, 16> BodyStmts;

  // TODO: case when kernel is functor
  // TODO: possible refactoring when functor case will be completed
  CXXRecordDecl *LC = getBodyAsLambda(KernelHelper);
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

    // Init Lambda fields
    llvm::SmallVector<Expr *, 16> InitCaptures;

    auto TargetFunc = dyn_cast<FunctionDecl>(DC);
    auto TargetFuncParam =
        TargetFunc->param_begin(); // Iterator to ParamVarDecl (VarDecl)
    for (auto Field : LC->fields()) {
      QualType ParamType = (*TargetFuncParam)->getOriginalType();
      auto DRE = DeclRefExpr::Create(
          S.Context, NestedNameSpecifierLoc(), SourceLocation(),
          *TargetFuncParam, false, DeclarationNameInfo(), ParamType, VK_LValue);

      CXXRecordDecl *CRD = Field->getType()->getAsCXXRecordDecl();
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

            // C++ address space attribute != opencl address space attribute
            Expr *qualifiersCast = ImplicitCastExpr::Create(
                S.Context, paramTy, CK_NoOp, DRE, nullptr, VK_LValue);
            Expr *Res =
                ImplicitCastExpr::Create(S.Context, paramTy, CK_LValueToRValue,
                                         qualifiersCast, nullptr, VK_RValue);

            ParamStmts.push_back(Res);

            // lambda.accessor.__set_pointer(kernel_parameter)
            CXXMemberCallExpr *Call = CXXMemberCallExpr::Create(
                S.Context, ME, ParamStmts, ResultTy, VK, SourceLocation());
            BodyStmts.push_back(Call);
          }
        }
      }
      TargetFuncParam++;
    }

    // In function from headers lambda is function parameter, we need
    // to replace all refs to this lambda with our vardecl.
    // I used TreeTransform here, but I'm not sure that it is good solution
    // Also I used map and I'm not sure about it too.
    Stmt* FunctionBody = KernelHelper->getBody();
    DeclMap DMap;
    ParmVarDecl* LambdaParam = *(KernelHelper->param_begin());
    // DeclRefExpr with valid source location but with decl which is not marked
    // as used is invalid.
    LambdaVD->setIsUsed();
    DMap[LambdaParam] = LambdaVD;
    // Without PushFunctionScope I had segfault. Maybe we also need to do pop.
    S.PushFunctionScope();
    KernelBodyTransform KBT(DMap, S);
    Stmt* NewBody = KBT.TransformStmt(FunctionBody).get();
    BodyStmts.push_back(NewBody);

  }
  return CompoundStmt::Create(S.Context, BodyStmts, SourceLocation(),
                              SourceLocation());
}

void BuildArgTys(ASTContext &Context,
                 llvm::SmallVector<DeclaratorDecl *, 16> &ArgDecls,
                 llvm::SmallVector<DeclaratorDecl *, 16> &NewArgDecls,
                 llvm::SmallVector<QualType, 16> &ArgTys) {
  for (auto V : ArgDecls) {
    QualType ArgTy = V->getType();
    QualType ActualArgType = ArgTy;
    StringRef Name = ArgTy.getBaseTypeIdentifier()->getName();
    // TODO: harden this check with additional validation that this class is
    // declared in cl::sycl namespace
    if (std::string(Name) == "accessor") {
      if (const auto *RecordDecl = ArgTy->getAsCXXRecordDecl()) {
        const auto *TemplateDecl =
            dyn_cast<ClassTemplateSpecializationDecl>(RecordDecl);
        if (TemplateDecl) {
          QualType PointeeType = TemplateDecl->getTemplateArgs()[0].getAsType();
          Qualifiers Quals = PointeeType.getQualifiers();
          // TODO: get address space from accessor template parameter.
          Quals.setAddressSpace(LangAS::opencl_global);
          PointeeType =
              Context.getQualifiedType(PointeeType.getUnqualifiedType(), Quals);
          QualType PointerType = Context.getPointerType(PointeeType);
          ActualArgType =
              Context.getQualifiedType(PointerType.getUnqualifiedType(), Quals);
        }
      }
    }
    DeclContext *DC = Context.getTranslationUnitDecl();

    IdentifierInfo *VarName = 0;
    SmallString<8> Str;
    llvm::raw_svector_ostream OS(Str);
    OS << "_arg_" << V->getIdentifier()->getName();
    VarName = &Context.Idents.get(OS.str());

    auto NewVarDecl = VarDecl::Create(
        Context, DC, SourceLocation(), SourceLocation(), VarName, ActualArgType,
        Context.getTrivialTypeSourceInfo(ActualArgType), SC_None);
    ArgTys.push_back(ActualArgType);
    NewArgDecls.push_back(NewVarDecl);
  }
}

void Sema::ConstructSYCLKernel(FunctionDecl *KernelHelper) {
  // TODO: Case when kernel is functor
  CXXRecordDecl *LE = getBodyAsLambda(KernelHelper);
  if (LE) {

    llvm::SmallVector<DeclaratorDecl *, 16> ArgDecls;

    for (const auto &V : LE->captures()) {
      ArgDecls.push_back(V.getCapturedVar());
    }

    llvm::SmallVector<QualType, 16> ArgTys;
    llvm::SmallVector<DeclaratorDecl *, 16> NewArgDecls;
    BuildArgTys(getASTContext(), ArgDecls, NewArgDecls, ArgTys);

    // Get Name for our kernel.
    const TemplateArgumentList *TemplateArgs =
        KernelHelper->getTemplateSpecializationArgs();
    QualType KernelNameType = TemplateArgs->get(0).getAsType();
    std::string Name = KernelNameType.getBaseTypeIdentifier()->getName().str();

    if (const auto *RecordDecl = KernelNameType->getAsCXXRecordDecl()) {
      const auto *TemplateDecl =
          dyn_cast<ClassTemplateSpecializationDecl>(RecordDecl);
      if (TemplateDecl) {
        QualType ParamType = TemplateDecl->getTemplateArgs()[0].getAsType();
        Name += "_" + ParamType.getAsString() + "_";
      }
    }

    FunctionDecl *SYCLKernel =
        CreateSYCLKernelFunction(getASTContext(), Name, ArgTys, NewArgDecls);

    CompoundStmt *SYCLKernelBody = CreateSYCLKernelBody(*this, KernelHelper, SYCLKernel);
    SYCLKernel->setBody(SYCLKernelBody);

    AddSyclKernel(SYCLKernel);
  }
}

