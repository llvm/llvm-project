//===- SemaSYCL.cpp - Semantic Analysis for SYCL constructs ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for SYCL constructs.
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaSYCL.h"
#include "TreeTransform.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/SYCLKernelInfo.h"
#include "clang/AST/StmtSYCL.h"
#include "clang/AST/TypeOrdering.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Sema/Attr.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"

using namespace clang;

// -----------------------------------------------------------------------------
// SYCL device specific diagnostics implementation
// -----------------------------------------------------------------------------

SemaSYCL::SemaSYCL(Sema &S) : SemaBase(S) {}

Sema::SemaDiagnosticBuilder SemaSYCL::DiagIfDeviceCode(SourceLocation Loc,
                                                       unsigned DiagID) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");
  FunctionDecl *FD = dyn_cast<FunctionDecl>(SemaRef.getCurLexicalContext());
  SemaDiagnosticBuilder::Kind DiagKind = [this, FD] {
    if (!FD)
      return SemaDiagnosticBuilder::K_Nop;
    if (SemaRef.getEmissionStatus(FD) == Sema::FunctionEmissionStatus::Emitted)
      return SemaDiagnosticBuilder::K_ImmediateWithCallStack;
    return SemaDiagnosticBuilder::K_Deferred;
  }();
  return SemaDiagnosticBuilder(DiagKind, Loc, DiagID, FD, SemaRef);
}

static bool isZeroSizedArray(SemaSYCL &S, QualType Ty) {
  if (const auto *CAT = S.getASTContext().getAsConstantArrayType(Ty))
    return CAT->isZeroSize();
  return false;
}

void SemaSYCL::deepTypeCheckForDevice(SourceLocation UsedAt,
                                      llvm::DenseSet<QualType> Visited,
                                      ValueDecl *DeclToCheck) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");
  // Emit notes only for the first discovered declaration of unsupported type
  // to avoid mess of notes. This flag is to track that error already happened.
  bool NeedToEmitNotes = true;

  auto Check = [&](QualType TypeToCheck, const ValueDecl *D) {
    bool ErrorFound = false;
    if (isZeroSizedArray(*this, TypeToCheck)) {
      DiagIfDeviceCode(UsedAt, diag::err_typecheck_zero_array_size) << 1;
      ErrorFound = true;
    }
    // Checks for other types can also be done here.
    if (ErrorFound) {
      if (NeedToEmitNotes) {
        if (auto *FD = dyn_cast<FieldDecl>(D))
          DiagIfDeviceCode(FD->getLocation(),
                           diag::note_illegal_field_declared_here)
              << FD->getType()->isPointerType() << FD->getType();
        else
          DiagIfDeviceCode(D->getLocation(), diag::note_declared_at);
      }
    }

    return ErrorFound;
  };

  // In case we have a Record used do the DFS for a bad field.
  SmallVector<const ValueDecl *, 4> StackForRecursion;
  StackForRecursion.push_back(DeclToCheck);

  // While doing DFS save how we get there to emit a nice set of notes.
  SmallVector<const FieldDecl *, 4> History;
  History.push_back(nullptr);

  do {
    const ValueDecl *Next = StackForRecursion.pop_back_val();
    if (!Next) {
      assert(!History.empty());
      // Found a marker, we have gone up a level.
      History.pop_back();
      continue;
    }
    QualType NextTy = Next->getType();

    if (!Visited.insert(NextTy).second)
      continue;

    auto EmitHistory = [&]() {
      // The first element is always nullptr.
      for (uint64_t Index = 1; Index < History.size(); ++Index) {
        DiagIfDeviceCode(History[Index]->getLocation(),
                         diag::note_within_field_of_type)
            << History[Index]->getType();
      }
    };

    if (Check(NextTy, Next)) {
      if (NeedToEmitNotes)
        EmitHistory();
      NeedToEmitNotes = false;
    }

    // In case pointer/array/reference type is met get pointee type, then
    // proceed with that type.
    while (NextTy->isAnyPointerType() || NextTy->isArrayType() ||
           NextTy->isReferenceType()) {
      if (NextTy->isArrayType())
        NextTy = QualType{NextTy->getArrayElementTypeNoTypeQual(), 0};
      else
        NextTy = NextTy->getPointeeType();
      if (Check(NextTy, Next)) {
        if (NeedToEmitNotes)
          EmitHistory();
        NeedToEmitNotes = false;
      }
    }

    if (const auto *RecDecl = NextTy->getAsRecordDecl()) {
      if (auto *NextFD = dyn_cast<FieldDecl>(Next))
        History.push_back(NextFD);
      // When nullptr is discovered, this means we've gone back up a level, so
      // the history should be cleaned.
      StackForRecursion.push_back(nullptr);
      llvm::append_range(StackForRecursion, RecDecl->fields());
    }
  } while (!StackForRecursion.empty());
}

ExprResult SemaSYCL::BuildUniqueStableNameExpr(SourceLocation OpLoc,
                                               SourceLocation LParen,
                                               SourceLocation RParen,
                                               TypeSourceInfo *TSI) {
  return SYCLUniqueStableNameExpr::Create(getASTContext(), OpLoc, LParen,
                                          RParen, TSI);
}

ExprResult SemaSYCL::ActOnUniqueStableNameExpr(SourceLocation OpLoc,
                                               SourceLocation LParen,
                                               SourceLocation RParen,
                                               ParsedType ParsedTy) {
  TypeSourceInfo *TSI = nullptr;
  QualType Ty = SemaRef.GetTypeFromParser(ParsedTy, &TSI);

  if (Ty.isNull())
    return ExprError();
  if (!TSI)
    TSI = getASTContext().getTrivialTypeSourceInfo(Ty, LParen);

  return BuildUniqueStableNameExpr(OpLoc, LParen, RParen, TSI);
}

void SemaSYCL::handleKernelAttr(Decl *D, const ParsedAttr &AL) {
  // The 'sycl_kernel' attribute applies only to function templates.
  const auto *FD = cast<FunctionDecl>(D);
  const FunctionTemplateDecl *FT = FD->getDescribedFunctionTemplate();
  assert(FT && "Function template is expected");

  // Function template must have at least two template parameters.
  const TemplateParameterList *TL = FT->getTemplateParameters();
  if (TL->size() < 2) {
    Diag(FT->getLocation(), diag::warn_sycl_kernel_num_of_template_params);
    return;
  }

  // Template parameters must be typenames.
  for (unsigned I = 0; I < 2; ++I) {
    const NamedDecl *TParam = TL->getParam(I);
    if (isa<NonTypeTemplateParmDecl>(TParam)) {
      Diag(FT->getLocation(),
           diag::warn_sycl_kernel_invalid_template_param_type);
      return;
    }
  }

  // Function must have at least one argument.
  if (getFunctionOrMethodNumParams(D) != 1) {
    Diag(FT->getLocation(), diag::warn_sycl_kernel_num_of_function_params);
    return;
  }

  // Function must return void.
  QualType RetTy = getFunctionOrMethodResultType(D);
  if (!RetTy->isVoidType()) {
    Diag(FT->getLocation(), diag::warn_sycl_kernel_return_type);
    return;
  }

  handleSimpleAttribute<DeviceKernelAttr>(*this, D, AL);
}

void SemaSYCL::handleKernelEntryPointAttr(Decl *D, const ParsedAttr &AL) {
  ParsedType PT = AL.getTypeArg();
  TypeSourceInfo *TSI = nullptr;
  (void)SemaRef.GetTypeFromParser(PT, &TSI);
  assert(TSI && "no type source info for attribute argument");
  D->addAttr(::new (SemaRef.Context)
                 SYCLKernelEntryPointAttr(SemaRef.Context, AL, TSI));
}

// Given a potentially qualified type, SourceLocationForUserDeclaredType()
// returns the source location of the canonical declaration of the unqualified
// desugared user declared type, if any. For non-user declared types, an
// invalid source location is returned. The intended usage of this function
// is to identify an appropriate source location, if any, for a
// "entity declared here" diagnostic note.
static SourceLocation SourceLocationForUserDeclaredType(QualType QT) {
  SourceLocation Loc;
  const Type *T = QT->getUnqualifiedDesugaredType();
  if (const TagType *TT = dyn_cast<TagType>(T))
    Loc = TT->getOriginalDecl()->getLocation();
  else if (const ObjCInterfaceType *ObjCIT = dyn_cast<ObjCInterfaceType>(T))
    Loc = ObjCIT->getDecl()->getLocation();
  return Loc;
}

static bool CheckSYCLKernelName(Sema &S, SourceLocation Loc,
                                QualType KernelName) {
  assert(!KernelName->isDependentType());

  if (!KernelName->isStructureOrClassType()) {
    // SYCL 2020 section 5.2, "Naming of kernels", only requires that the
    // kernel name be a C++ typename. However, the definition of "kernel name"
    // in the glossary states that a kernel name is a class type. Neither
    // section explicitly states whether the kernel name type can be
    // cv-qualified. For now, kernel name types are required to be class types
    // and that they may be cv-qualified. The following issue requests
    // clarification from the SYCL WG.
    //   https://github.com/KhronosGroup/SYCL-Docs/issues/568
    S.Diag(Loc, diag::warn_sycl_kernel_name_not_a_class_type) << KernelName;
    SourceLocation DeclTypeLoc = SourceLocationForUserDeclaredType(KernelName);
    if (DeclTypeLoc.isValid())
      S.Diag(DeclTypeLoc, diag::note_entity_declared_at) << KernelName;
    return true;
  }

  return false;
}

void SemaSYCL::CheckSYCLExternalFunctionDecl(FunctionDecl *FD) {
  const auto *SEAttr = FD->getAttr<SYCLExternalAttr>();
  assert(SEAttr && "Missing sycl_external attribute");
  if (!FD->isInvalidDecl() && !FD->isTemplated()) {
    if (!FD->isExternallyVisible())
      if (!FD->isFunctionTemplateSpecialization() ||
          FD->getTemplateSpecializationInfo()->isExplicitSpecialization())
        Diag(SEAttr->getLocation(), diag::err_sycl_external_invalid_linkage)
            << SEAttr;
  }
  if (FD->isDeletedAsWritten()) {
    Diag(SEAttr->getLocation(),
         diag::err_sycl_external_invalid_deleted_function)
        << SEAttr;
  }
}

void SemaSYCL::CheckSYCLEntryPointFunctionDecl(FunctionDecl *FD) {
  // Ensure that all attributes present on the declaration are consistent
  // and warn about any redundant ones.
  SYCLKernelEntryPointAttr *SKEPAttr = nullptr;
  for (auto *SAI : FD->specific_attrs<SYCLKernelEntryPointAttr>()) {
    if (!SKEPAttr) {
      SKEPAttr = SAI;
      continue;
    }
    if (!getASTContext().hasSameType(SAI->getKernelName(),
                                     SKEPAttr->getKernelName())) {
      Diag(SAI->getLocation(), diag::err_sycl_entry_point_invalid_redeclaration)
          << SKEPAttr << SAI->getKernelName() << SKEPAttr->getKernelName();
      Diag(SKEPAttr->getLocation(), diag::note_previous_attribute);
      SAI->setInvalidAttr();
    } else {
      Diag(SAI->getLocation(),
           diag::warn_sycl_entry_point_redundant_declaration)
          << SAI;
      Diag(SKEPAttr->getLocation(), diag::note_previous_attribute);
    }
  }
  assert(SKEPAttr && "Missing sycl_kernel_entry_point attribute");

  // Ensure the kernel name type is valid.
  if (!SKEPAttr->getKernelName()->isDependentType() &&
      CheckSYCLKernelName(SemaRef, SKEPAttr->getLocation(),
                          SKEPAttr->getKernelName()))
    SKEPAttr->setInvalidAttr();

  // Ensure that an attribute present on the previous declaration
  // matches the one on this declaration.
  FunctionDecl *PrevFD = FD->getPreviousDecl();
  if (PrevFD && !PrevFD->isInvalidDecl()) {
    const auto *PrevSKEPAttr = PrevFD->getAttr<SYCLKernelEntryPointAttr>();
    if (PrevSKEPAttr && !PrevSKEPAttr->isInvalidAttr()) {
      if (!getASTContext().hasSameType(SKEPAttr->getKernelName(),
                                       PrevSKEPAttr->getKernelName())) {
        Diag(SKEPAttr->getLocation(),
             diag::err_sycl_entry_point_invalid_redeclaration)
            << SKEPAttr << SKEPAttr->getKernelName()
            << PrevSKEPAttr->getKernelName();
        Diag(PrevSKEPAttr->getLocation(), diag::note_previous_decl) << PrevFD;
        SKEPAttr->setInvalidAttr();
      }
    }
  }

  if (isa<CXXConstructorDecl>(FD)) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << SKEPAttr << /*constructor*/ 3;
    SKEPAttr->setInvalidAttr();
  }
  if (isa<CXXDestructorDecl>(FD)) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << SKEPAttr << /*destructor*/ 4;
    SKEPAttr->setInvalidAttr();
  }

  if (FD->isVariadic()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << SKEPAttr << /*variadic function*/ 0;
    SKEPAttr->setInvalidAttr();
  }

  if (FD->isDefaulted()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << SKEPAttr << /*defaulted function*/ 2;
    SKEPAttr->setInvalidAttr();
  } else if (FD->isDeleted()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << SKEPAttr << /*deleted function*/ 1;
    SKEPAttr->setInvalidAttr();
  }

  if (FD->isConsteval()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << SKEPAttr << /*consteval function*/ 7;
    SKEPAttr->setInvalidAttr();
  } else if (FD->isConstexpr()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << SKEPAttr << /*constexpr function*/ 6;
    SKEPAttr->setInvalidAttr();
  }

  if (FD->isNoReturn()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << SKEPAttr << /*function declared with the 'noreturn' attribute*/ 8;
    SKEPAttr->setInvalidAttr();
  }

  if (FD->getReturnType()->isUndeducedType()) {
    Diag(SKEPAttr->getLocation(),
         diag::err_sycl_entry_point_deduced_return_type)
        << SKEPAttr;
    SKEPAttr->setInvalidAttr();
  } else if (!FD->getReturnType()->isDependentType() &&
             !FD->getReturnType()->isVoidType()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_return_type)
        << SKEPAttr;
    SKEPAttr->setInvalidAttr();
  }

  if (!FD->isInvalidDecl() && !FD->isTemplated() &&
      !SKEPAttr->isInvalidAttr()) {
    const SYCLKernelInfo *SKI =
        getASTContext().findSYCLKernelInfo(SKEPAttr->getKernelName());
    if (SKI) {
      if (!declaresSameEntity(FD, SKI->getKernelEntryPointDecl())) {
        // FIXME: This diagnostic should include the origin of the kernel
        // FIXME: names; not just the locations of the conflicting declarations.
        Diag(FD->getLocation(), diag::err_sycl_kernel_name_conflict)
            << SKEPAttr;
        Diag(SKI->getKernelEntryPointDecl()->getLocation(),
             diag::note_previous_declaration);
        SKEPAttr->setInvalidAttr();
      }
    } else {
      getASTContext().registerSYCLEntryPointFunction(FD);
    }
  }
}

ExprResult SemaSYCL::BuildSYCLKernelLaunchIdExpr(FunctionDecl *FD,
                                                 QualType KNT) {

  ASTContext &Ctx = SemaRef.getASTContext();
  // Some routines need a valid source location to work correctly.
  SourceLocation BodyLoc =
      FD->getEndLoc().isValid() ? FD->getEndLoc() : FD->getLocation();

  IdentifierInfo &LaunchFooName =
      Ctx.Idents.get("sycl_kernel_launch", tok::TokenKind::identifier);

  // Perform overload resolution for a call to an accessible (member) function
  // template named 'sycl_kernel_launch' "from within the definition of
  // FD where":
  // - The kernel name type is passed as the first template argument.
  // - Any remaining template parameters are deduced from the function
  //   arguments or assigned by default template arguments.
  // - 'this' is passed as the implicit function argument if 'FD' is a
  //   non-static member function.
  // - The name of the kernel, expressed as a string literal, is passed as the
  //   first function argument.
  // - The parameters of FD are forwarded as-if by 'std::forward()' as the
  //   remaining explicit function arguments.
  // - Any remaining function arguments are initialized by default arguments.
  LookupResult Result(SemaRef, &LaunchFooName, BodyLoc,
                      Sema::LookupOrdinaryName);
  CXXScopeSpec SS;
  SemaRef.LookupTemplateName(Result, SemaRef.getCurScope(), SS,
                             /*ObjectType=*/QualType(),
                             /*EnteringContext=*/false, BodyLoc);

  if (Result.empty() || Result.isAmbiguous()) {
    SemaRef.Diag(BodyLoc, SemaRef.getLangOpts().SYCLIsHost
                              ? diag::err_sycl_host_no_launch_function
                              : diag::warn_sycl_device_no_host_launch_function);
    SemaRef.Diag(BodyLoc, diag::note_sycl_host_launch_function);

    return ExprError();
  }

  TemplateArgumentListInfo TALI{BodyLoc, BodyLoc};
  TemplateArgument KNTA = TemplateArgument(KNT);
  TemplateArgumentLoc TAL =
      SemaRef.getTrivialTemplateArgumentLoc(KNTA, QualType(), BodyLoc);
  TALI.addArgument(TAL);
  ExprResult IdExpr;
  if (SemaRef.isPotentialImplicitMemberAccess(SS, Result,
                                              /*IsAddressOfOperand=*/false))
    // BuildPossibleImplicitMemberExpr creates UnresolvedMemberExpr. Using it
    // allows to pass implicit/explicit this argument automatically.
    IdExpr = SemaRef.BuildPossibleImplicitMemberExpr(SS, BodyLoc, Result, &TALI,
                                                     SemaRef.getCurScope());
  else
    IdExpr = SemaRef.BuildTemplateIdExpr(SS, BodyLoc, Result,
                                         /*RequiresADL=*/true, &TALI);

  // Can happen if SKEP attributed function is a static member, but the launcher
  // is a regular member. Perhaps emit a note saying that we're in host code
  // synthesis.
  if (IdExpr.isInvalid())
    return ExprError();

  return IdExpr;
}

StmtResult SemaSYCL::BuildUnresolvedSYCLKernelCallStmt(CompoundStmt *CS,
                                                             Expr *IdExpr) {
  return UnresolvedSYCLKernelCallStmt::Create(SemaRef.getASTContext(), CS,
                                                    IdExpr);
}

namespace {

void PrepareKernelArgumentsForKernelLaunch(SmallVectorImpl<Expr *> &Args,
                                                  const SYCLKernelInfo *SKI,
                                                  Sema &SemaRef,
                                                  SourceLocation Loc) {
  assert(SKI && "Need a kernel!");
  ASTContext &Ctx = SemaRef.getASTContext();

  // Prepare a string literal that contains the kernel name.
  const std::string KernelName = SKI->GetKernelName();
  QualType KernelNameCharTy = Ctx.CharTy.withConst();
  llvm::APInt KernelNameSize(Ctx.getTypeSize(Ctx.getSizeType()),
                             KernelName.size() + 1);
  QualType KernelNameArrayTy = Ctx.getConstantArrayType(
      KernelNameCharTy, KernelNameSize, nullptr, ArraySizeModifier::Normal, 0);
  Expr *KernelNameExpr =
      StringLiteral::Create(Ctx, KernelName, StringLiteralKind::Ordinary,
                            /*Pascal*/ false, KernelNameArrayTy, Loc);
  Args.push_back(KernelNameExpr);

  // Right now we simply forward the arguments of the skep-attributed function.
  // With decomposition present there can be another logic.
  // Make sure to use CurContext to avoid diagnostics that we're using a
  // variable coming from another context. The function should be the same as in
  // the kernel info though.
  auto *FD = cast<FunctionDecl>(SemaRef.CurContext);
  assert(declaresSameEntity(FD, SKI->getKernelEntryPointDecl()));
  for (ParmVarDecl *PVD : FD->parameters()) {
    QualType ParamType = PVD->getOriginalType().getNonReferenceType();
    Expr *DRE = SemaRef.BuildDeclRefExpr(PVD, ParamType, VK_LValue, Loc);
    assert(DRE);
    Args.push_back(DRE);
  }
}

StmtResult BuildSYCLKernelLaunchStmt(Sema &SemaRef,
                                            const SYCLKernelInfo *SKI,
                                            Expr *IdExpr, SourceLocation Loc) {
  SmallVector<Stmt *> Stmts;
  assert(SKI && "Need a Kernel!");

  if (IdExpr) {
    llvm::SmallVector<Expr *, 12> Args;
    PrepareKernelArgumentsForKernelLaunch(Args, SKI, SemaRef, Loc);
    ExprResult LaunchResult =
        SemaRef.BuildCallExpr(SemaRef.getCurScope(), IdExpr, Loc, Args, Loc);
    if (LaunchResult.isInvalid())
      return StmtError();

    Stmts.push_back(LaunchResult.get());
  }

  return CompoundStmt::Create(SemaRef.getASTContext(), Stmts,
                              FPOptionsOverride(), Loc, Loc);
}

// The body of a function declared with the [[sycl_kernel_entry_point]]
// attribute is cloned and transformed to substitute references to the original
// function parameters with references to replacement variables that stand in
// for SYCL kernel parameters or local variables that reconstitute a decomposed
// SYCL kernel argument.
class OutlinedFunctionDeclBodyInstantiator
    : public TreeTransform<OutlinedFunctionDeclBodyInstantiator> {
public:
  using ParmDeclMap = llvm::DenseMap<ParmVarDecl *, VarDecl *>;

  OutlinedFunctionDeclBodyInstantiator(Sema &S, ParmDeclMap &M,
                                       FunctionDecl *FD)
      : TreeTransform<OutlinedFunctionDeclBodyInstantiator>(S), SemaRef(S),
        MapRef(M), FD(FD) {}

  // A new set of AST nodes is always required.
  bool AlwaysRebuild() { return true; }

  // Transform ParmVarDecl references to the supplied replacement variables.
  ExprResult TransformDeclRefExpr(DeclRefExpr *DRE) {
    const ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(DRE->getDecl());
    if (PVD) {
      ParmDeclMap::iterator I = MapRef.find(PVD);
      if (I != MapRef.end()) {
        VarDecl *VD = I->second;
        assert(SemaRef.getASTContext().hasSameUnqualifiedType(PVD->getType(),
                                                              VD->getType()));
        assert(!VD->getType().isMoreQualifiedThan(PVD->getType(),
                                                  SemaRef.getASTContext()));
        VD->setIsUsed();
        return DeclRefExpr::Create(
            SemaRef.getASTContext(), DRE->getQualifierLoc(),
            DRE->getTemplateKeywordLoc(), VD, false, DRE->getNameInfo(),
            DRE->getType(), DRE->getValueKind());
      }
    }
    return DRE;
  }

  // Diagnose CXXThisExpr in a potentially evaluated expression.
  ExprResult TransformCXXThisExpr(CXXThisExpr *CTE) {
    if (SemaRef.currentEvaluationContext().isPotentiallyEvaluated()) {
      SemaRef.Diag(CTE->getExprLoc(), diag::err_sycl_entry_point_invalid_this)
          << (CTE->isImplicitCXXThis() ? /* implicit */ 1 : /* empty */ 0)
          << FD->getAttr<SYCLKernelEntryPointAttr>();
    }
    return CTE;
  }

private:
  Sema &SemaRef;
  ParmDeclMap &MapRef;
  FunctionDecl *FD;
};

OutlinedFunctionDecl *BuildSYCLKernelEntryPointOutline(Sema &SemaRef,
                                                       FunctionDecl *FD,
                                                       CompoundStmt *Body) {
  using ParmDeclMap = OutlinedFunctionDeclBodyInstantiator::ParmDeclMap;
  ParmDeclMap ParmMap;

  OutlinedFunctionDecl *OFD = OutlinedFunctionDecl::Create(
      SemaRef.getASTContext(), FD, FD->getNumParams());
  unsigned i = 0;
  for (ParmVarDecl *PVD : FD->parameters()) {
    ImplicitParamDecl *IPD = ImplicitParamDecl::Create(
        SemaRef.getASTContext(), OFD, SourceLocation(), PVD->getIdentifier(),
        PVD->getType(), ImplicitParamKind::Other);
    OFD->setParam(i, IPD);
    ParmMap[PVD] = IPD;
    ++i;
  }

  OutlinedFunctionDeclBodyInstantiator OFDBodyInstantiator(SemaRef, ParmMap,
                                                           FD);
  Stmt *OFDBody = OFDBodyInstantiator.TransformStmt(Body).get();
  OFD->setBody(OFDBody);
  OFD->setNothrow();

  return OFD;
}

} // unnamed namespace

StmtResult SemaSYCL::BuildSYCLKernelCallStmt(FunctionDecl *FD,
                                             CompoundStmt *Body,
                                             Expr *LaunchIdExpr) {
  assert(!FD->isInvalidDecl());
  assert(!FD->isTemplated());
  assert(FD->hasPrototype());
  // The current context must be the function definition context to ensure
  // that name lookup and parameter and local variable creation are performed
  // within the correct scope.
  assert(SemaRef.CurContext == FD);

  const auto *SKEPAttr = FD->getAttr<SYCLKernelEntryPointAttr>();
  assert(SKEPAttr && "Missing sycl_kernel_entry_point attribute");
  assert(!SKEPAttr->isInvalidAttr() &&
         "sycl_kernel_entry_point attribute is invalid");

  // Ensure that the kernel name was previously registered and that the
  // stored declaration matches.
  const SYCLKernelInfo &SKI =
      getASTContext().getSYCLKernelInfo(SKEPAttr->getKernelName());
  assert(declaresSameEntity(SKI.getKernelEntryPointDecl(), FD) &&
         "SYCL kernel name conflict");

  // Build the outline of the synthesized device entry point function.
  OutlinedFunctionDecl *OFD =
      BuildSYCLKernelEntryPointOutline(SemaRef, FD, Body);
  assert(OFD);

  // Build host kernel launch stmt.
  SourceLocation BodyLoc =
      FD->getEndLoc().isValid() ? FD->getEndLoc() : FD->getLocation();
  StmtResult LaunchRes =
      BuildSYCLKernelLaunchStmt(SemaRef, &SKI, LaunchIdExpr, BodyLoc);

  Stmt *NewBody =
      new (getASTContext()) SYCLKernelCallStmt(Body, LaunchRes.get(), OFD);

  return NewBody;
}
