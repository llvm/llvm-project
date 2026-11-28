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
#include "clang/AST/SubobjectVisitor.h"
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
         "Device diagnostics Should only be issued during device compilation");
  SemaDiagnosticBuilder::Kind DiagKind = SemaDiagnosticBuilder::K_Nop;
  FunctionDecl *FD = SemaRef.getCurFunctionDecl(/*AllowLambda=*/true);
  if (FD) {
    Sema::FunctionEmissionStatus FES = SemaRef.getEmissionStatus(FD);
    switch (FES) {
    case Sema::FunctionEmissionStatus::Emitted:
      DiagKind = SemaDiagnosticBuilder::K_ImmediateWithCallStack;
      break;
    case Sema::FunctionEmissionStatus::Unknown:
    case Sema::FunctionEmissionStatus::TemplateDiscarded:
      DiagKind = SemaDiagnosticBuilder::K_Deferred;
      break;
    case Sema::FunctionEmissionStatus::OMPDiscarded:
      llvm_unreachable("OMPDiscarded unexpected in SYCL device compilation");
    case Sema::FunctionEmissionStatus::CUDADiscarded:
      llvm_unreachable("CUDADiscarded unexpected in SYCL device compilation");
    }
  }
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

  handleSimpleAttribute<SYCLKernelAttr>(*this, D, AL);
}

void SemaSYCL::handleKernelEntryPointAttr(Decl *D, const ParsedAttr &AL) {
  ParsedType PT = AL.getTypeArg();
  TypeSourceInfo *TSI = nullptr;
  (void)SemaRef.GetTypeFromParser(PT, &TSI);
  assert(TSI && "no type source info for attribute argument");
  D->addAttr(::new (SemaRef.Context)
                 SYCLKernelEntryPointAttr(SemaRef.Context, AL, TSI));
}

void SemaSYCL::CheckDeviceUseOfDecl(NamedDecl *ND, SourceLocation Loc) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL device compilation");

  // Function declarations with the sycl_kernel_entry_point attribute cannot
  // be ODR-used in a potentially evaluated context.
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(ND)) {
    if (const auto *SKEPAttr = FD->getAttr<SYCLKernelEntryPointAttr>()) {
      if (SemaRef.currentEvaluationContext().isPotentiallyEvaluated()) {
        DiagIfDeviceCode(Loc, diag::err_sycl_entry_point_device_use)
            << FD << SKEPAttr;
        DiagIfDeviceCode(SKEPAttr->getLocation(), diag::note_attribute) << FD;
      }
    }
  }
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
    Loc = TT->getDecl()->getLocation();
  else if (const auto *ObjCIT = dyn_cast<ObjCInterfaceType>(T))
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

void SemaSYCL::handleSYCLExternalAttr(Decl *D, const ParsedAttr &AL) {
  // SYCL 2020 section 5.4 prohibits calling a variadic function from device
  // code, so the attribute cannot be honored; declaring one is not prohibited,
  // so ignore the attribute rather than reject the declaration.
  if (cast<FunctionDecl>(D)->isVariadic()) {
    Diag(AL.getLoc(), diag::warn_sycl_external_ignored_variadic_function) << AL;
    return;
  }

  handleSimpleAttribute<SYCLExternalAttr>(*this, D, AL);
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
        << SKEPAttr << diag::InvalidSKEPReason::Constructor;
    SKEPAttr->setInvalidAttr();
  }
  if (isa<CXXDestructorDecl>(FD)) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << SKEPAttr << diag::InvalidSKEPReason::Destructor;
    SKEPAttr->setInvalidAttr();
  }
  if (const auto *MD = dyn_cast<CXXMethodDecl>(FD)) {
    if (MD->isExplicitObjectMemberFunction()) {
      Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
          << SKEPAttr << diag::InvalidSKEPReason::ExplicitObjectFn;
      SKEPAttr->setInvalidAttr();
    }
  }

  if (FD->isVariadic()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << SKEPAttr << diag::InvalidSKEPReason::VariadicFn;
    SKEPAttr->setInvalidAttr();
  }

  if (FD->isDefaulted()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << SKEPAttr << diag::InvalidSKEPReason::DefaultedFn;
    SKEPAttr->setInvalidAttr();
  } else if (FD->isDeleted()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << SKEPAttr << diag::InvalidSKEPReason::DeletedFn;
    SKEPAttr->setInvalidAttr();
  }

  if (FD->isConsteval()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << SKEPAttr << diag::InvalidSKEPReason::ConstevalFn;
    SKEPAttr->setInvalidAttr();
  } else if (FD->isConstexpr()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << SKEPAttr << diag::InvalidSKEPReason::ConstexprFn;
    SKEPAttr->setInvalidAttr();
  }

  if (FD->isNoReturn()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << SKEPAttr << diag::InvalidSKEPReason::NoreturnFn;
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

ClassTemplateDecl *SemaSYCL::GetSYCLKernelInfoClassTemplate() {
  if (!SYCLKernelInfoClassTemplate) {
    ASTContext &Ctx = getASTContext();
    SourceLocation Loc;

    // The class template is implicitly declared in the translation unit
    // declaration context.
    DeclContext *DC = Ctx.getTranslationUnitDecl();

    // Build a reserved identifier for the name of the class template. The
    // name is formally unspecified.
    IdentifierInfo *KernelInfoID =
        &Ctx.Idents.get("__sycl_kernel_info", tok::TokenKind::identifier);

    // Build a template parameter list consisting of a single type template
    // parameter corresponding to a SYCL kernel name type. A reserved
    // identifier is used for the formally unspecified name of the template
    // parameter.
    IdentifierInfo *KernelNameID =
        &Ctx.Idents.get("__sycl_kernel_name", tok::TokenKind::identifier);
    TemplateTypeParmDecl *KernelNameParam = TemplateTypeParmDecl::Create(
        Ctx, DC, /*KeyLoc*/ Loc, /*NameLoc*/ Loc, 0, 0, KernelNameID,
        /*Typename*/ true, /*ParameterPack*/ false);
    TemplateParameterList *ParamList = TemplateParameterList::Create(
        Ctx, /*TemplateLoc*/ Loc, /*LAngleLoc*/ Loc, KernelNameParam,
        /*RAngleLoc*/ Loc, /*RequiresClause*/ nullptr);

    // Build the class template and add it to its declaration context.
    CXXRecordDecl *KernelInfoDecl =
        CXXRecordDecl::Create(Ctx, TagTypeKind::Class, DC, /*StartLoc*/ Loc,
                              /*IdLoc*/ Loc, KernelInfoID);
    SYCLKernelInfoClassTemplate = ClassTemplateDecl::Create(
        Ctx, DC, /*L*/ Loc, DeclarationName(KernelInfoID), ParamList,
        KernelInfoDecl);
    KernelInfoDecl->setDescribedClassTemplate(SYCLKernelInfoClassTemplate);
    DC->addDecl(SYCLKernelInfoClassTemplate);
    SemaRef.addSynthesizedTopLevelDecl(SYCLKernelInfoClassTemplate);
  }
  return SYCLKernelInfoClassTemplate;
}

QualType SemaSYCL::GetSYCLKernelInfoClassSpecializationType(QualType KNT) {
  ASTContext &Ctx = getASTContext();

  // Retrieve the class template for which a specialization is requested.
  ClassTemplateDecl *KernelInfoTemplateDecl = GetSYCLKernelInfoClassTemplate();

  // Build the template argument for the requested specialization.
  TemplateArgument KNTA = TemplateArgument(KNT);

  // If KNT is a dependent type, then a canonical type without an underlying
  // type is returned. Otherwise, an explicit specialization declaration for
  // a canonical underlying type is found or created and a canonical type
  // returned for it.
  QualType UnderlyingKernelInfoType;
  if (!KNT->isDependentType()) {
    // KNT is not a dependent type, check for a previously synthesized explicit
    // specialization declaration.
    CanQualType CanonicalKNT = Ctx.getCanonicalType(KNT);
    auto IT = SYCLKernelInfoClassTemplateSpecializations.find(CanonicalKNT);
    if (IT == SYCLKernelInfoClassTemplateSpecializations.end()) {
      // No previous explicit specialization declaration found; synthesize one.
      // KNT and KNTA are reassigned to avoid inadvertent use of a non-canonical
      // type.
      KNT = CanonicalKNT;
      KNTA = TemplateArgument(KNT);

      // Find the insertion position for the new specialization and ensure no
      // such specialization already exists.
      llvm::FoldingSetInsertToken InsertToken;
      ClassTemplateSpecializationDecl *KernelInfoDecl =
          KernelInfoTemplateDecl->findSpecialization(KNTA, InsertToken);
      assert(KernelInfoDecl == nullptr &&
             "__sycl_kernel_info specialization unexpectedly already exists");

      // Declare the explicit specialization, register it with the class
      // template, and add it to the corresponding DeclContext.
      DeclContext *DC = KernelInfoTemplateDecl->getDeclContext();
      SourceLocation Loc;
      KernelInfoDecl = ClassTemplateSpecializationDecl::Create(
          Ctx, KernelInfoTemplateDecl->getTemplatedDecl()->getTagKind(), DC,
          /*StartLoc*/ Loc, /*IdLoc*/ Loc, KernelInfoTemplateDecl, KNTA,
          /*StrictPackMatch*/ false, /*PrevDecl*/ nullptr);
      KernelInfoDecl->setSpecializationKind(TSK_ExplicitSpecialization);
      KernelInfoTemplateDecl->AddSpecialization(KernelInfoDecl, InsertToken);
      DC->addDecl(KernelInfoDecl);
      SemaRef.addSynthesizedTopLevelDecl(KernelInfoDecl);

      // Cache the new declaration for subsequent calls.
      SYCLKernelInfoClassTemplateSpecializations.insert(
          std::make_pair(CanonicalKNT, KernelInfoDecl));

      // Retrieve the canonical underlying type to use for the returned
      // specialization type.
      UnderlyingKernelInfoType = Ctx.getCanonicalTagType(KernelInfoDecl);
    } else {
      UnderlyingKernelInfoType = Ctx.getCanonicalTagType(IT->second);
    }
  }

  QualType KernelInfoType = Ctx.getTemplateSpecializationType(
      ElaboratedTypeKeyword::None, TemplateName(KernelInfoTemplateDecl), KNTA,
      /*CanonicalArgs*/ {}, UnderlyingKernelInfoType);

  return KernelInfoType;
}

namespace {

/// BuildSYCLKernelInfoClassSpecializationDefn builds an explicit
/// specialization definition for the SYCL kernel info class template
/// specialized for 'KNT'. 'KNT' cannot be a dependent type. The
/// synthesized definition has the following structure where
/// '<kernel-name>' represents the generated name of the offload
/// kernel entry point function.
///
/// template<>
/// struct  __sycl_kernel_info<KNT> {
///   using kernel_name = KNT;
///   static constexpr const char kernel_entry_point_name[] = "<kernel-name>";
/// };
void BuildSYCLKernelInfoClassSpecializationDefn(Sema &SemaRef, QualType KNT) {
  assert(!KNT->isDependentType());

  ASTContext &Ctx = SemaRef.getASTContext();

  // Retrieve the class template to be specialized.
  ClassTemplateDecl *KernelInfoTemplateDecl =
      SemaRef.SYCL().GetSYCLKernelInfoClassTemplate();

  // Retrieve the specialization declaration that should have already been
  // generated and validate that it corresponds to an explicit specialization
  // as opposed to a somehow instantiated implicit specialization.
  TemplateArgument KNTA = TemplateArgument(KNT);
  llvm::FoldingSetInsertToken InsertToken;
  ClassTemplateSpecializationDecl *KernelInfoDecl =
      KernelInfoTemplateDecl->findSpecialization(KNTA, InsertToken);
  assert(KernelInfoDecl != nullptr &&
         "__sycl_kernel_info specialization unexpectedly not found");
  assert(KernelInfoDecl->getSpecializationKind() == TSK_ExplicitSpecialization);

  // Start defining the specialization.
  KernelInfoDecl->startDefinition();

  // An injected class declaration would normally be declared for an explicit
  // specialization, but isn't needed, so is skipped.

  // Build the kernel_name type alias member.
  SourceLocation Loc;
  TypeSourceInfo *TSI = Ctx.getTrivialTypeSourceInfo(KNT, Loc);
  IdentifierInfo &KernelNameTypeAliasID =
      Ctx.Idents.get("kernel_name", tok::TokenKind::identifier);
  TypeAliasDecl *KernelNameTypeAliasDecl = TypeAliasDecl::Create(
      Ctx, KernelInfoDecl, Loc, Loc, &KernelNameTypeAliasID, TSI);
  KernelNameTypeAliasDecl->setAccess(AS_public);
  KernelInfoDecl->addDecl(KernelNameTypeAliasDecl);

  // Build the kernel_entry_point_name static data member with its initializer.
  const SYCLKernelInfo &SKI = Ctx.getSYCLKernelInfo(KNT);
  const std::string &KernelName = SKI.GetKernelName();
  QualType KernelNameCharTy = Ctx.CharTy;
  QualType KernelNameArrayTy =
      Ctx.getStringLiteralArrayType(KernelNameCharTy, KernelName.size());
  IdentifierInfo &KernelEntryPointNameID =
      Ctx.Idents.get("kernel_entry_point_name", tok::TokenKind::identifier);
  VarDecl *KernelEntryPointNameDecl = VarDecl::Create(
      Ctx, KernelInfoDecl, Loc, Loc, &KernelEntryPointNameID, KernelNameArrayTy,
      Ctx.getTrivialTypeSourceInfo(KernelNameArrayTy, Loc), SC_Static);
  Expr *KernelNameExpr =
      StringLiteral::Create(Ctx, KernelName, StringLiteralKind::Ordinary,
                            /*Pascal*/ false, KernelNameArrayTy, Loc);
  KernelEntryPointNameDecl->setConstexpr(true);
  KernelEntryPointNameDecl->setImplicitlyInline();
  KernelEntryPointNameDecl->setAccess(AS_public);
  KernelEntryPointNameDecl->setInitStyle(VarDecl::CInit);
  SemaRef.AddInitializerToDecl(KernelEntryPointNameDecl, KernelNameExpr,
                               /*DirectInit*/ false);
  KernelInfoDecl->addDecl(KernelEntryPointNameDecl);

  // Complete the definition of the specialization.
  KernelInfoDecl->completeDefinition();
}

/// BuildSYCLKernelLaunchIdExpr performs lookup for a 'sycl_kernel_launch'
/// template with 'KIT' used as an explicit type template argument. If
/// lookup fails, a diagnostic is issued and an error expression returned.
/// Otherwise, an UnresolvedLookupExpr, UnresolvedMemberExpr, DeclRefExpr,
/// or MemberExpr will be returned. The resulting expression is intended
/// to be passed as the 'IdExpr' argument to BuildSYCLKernelLaunchCallStmt()
/// after the function body has been parsed.
///
/// 'FD' must be a function declared with a valid sycl_kernel_entry_point
/// attribute and the current Sema context must match 'FD' so that scopes
/// are properly in place such that lookup is performed as if from the first
/// statement of the function body.
///
/// 'KIT' is the SYCL kernel info type to be used for lookup and it may
/// be a dependent type.
ExprResult BuildSYCLKernelLaunchIdExpr(Sema &SemaRef, FunctionDecl *FD,
                                       QualType KIT) {
  // The current context must be the function definition context to ensure
  // that name lookup is performed within the correct scope.
  assert(SemaRef.CurContext == FD && "The current declaration context does not "
                                     "match the requested function context");

  // An appropriate source location is required to emit diagnostics if
  // lookup fails to produce an overload set. The desired location is the
  // start of the function body, but that is not yet available since the
  // body of the function has not yet been set when this function is called.
  // The general location of the function is used instead.
  SourceLocation Loc = FD->getLocation();

  ASTContext &Ctx = SemaRef.getASTContext();
  IdentifierInfo &SYCLKernelLaunchID =
      Ctx.Idents.get("sycl_kernel_launch", tok::TokenKind::identifier);

  // Establish a code synthesis context for the implicit name lookup of
  // a template named 'sycl_kernel_launch'. In the event of an error, this
  // ensures an appropriate diagnostic note is issued to explain why the
  // lookup was performed.
  Sema::CodeSynthesisContext CSC;
  CSC.Kind = Sema::CodeSynthesisContext::SYCLKernelLaunchLookup;
  CSC.Entity = FD;
  Sema::ScopedCodeSynthesisContext ScopedCSC(SemaRef, CSC);

  // Perform ordinary name lookup for a function or variable template that
  // accepts a single type template argument.
  LookupResult Result(SemaRef, &SYCLKernelLaunchID, Loc,
                      Sema::LookupOrdinaryName);
  CXXScopeSpec EmptySS;
  if (SemaRef.LookupTemplateName(Result, SemaRef.getCurScope(), EmptySS,
                                 /*ObjectType*/ QualType(),
                                 /*EnteringContext*/ false,
                                 Sema::TemplateNameIsRequired))
    return ExprError();
  if (Result.isAmbiguous())
    return ExprError();

  TemplateArgumentListInfo TALI{Loc, Loc};
  TemplateArgument KITA = TemplateArgument(KIT);
  TemplateArgumentLoc TAL =
      SemaRef.getTrivialTemplateArgumentLoc(KITA, QualType(), Loc);
  TALI.addArgument(TAL);

  ExprResult IdExpr;
  if (SemaRef.isPotentialImplicitMemberAccess(EmptySS, Result,
                                              /*IsAddressOfOperand*/ false)) {
    // The lookup result allows for a possible implicit member access that
    // would require an implicit or explicit 'this' argument.
    IdExpr = SemaRef.BuildPossibleImplicitMemberExpr(
        EmptySS, SourceLocation(), Result, &TALI, SemaRef.getCurScope());
  } else {
    IdExpr = SemaRef.BuildTemplateIdExpr(EmptySS, SourceLocation(), Result,
                                         /*RequiresADL*/ true, &TALI);
  }

  // The resulting expression may be invalid if, for example, 'FD' is a
  // non-static member function and sycl_kernel_launch lookup selects a
  // member function (which would require a 'this' argument which is
  // not available).
  if (IdExpr.isInvalid())
    return ExprError();

  return IdExpr;
}

/// BuildSYCLReifyKernelParamsIdExpr performs lookup for a
/// 'sycl_reify_kernel_parameters' template with 'KIT' used as an explicit
/// type template argument. If lookup fails, no diagnostic is issued, but
/// an error expression is returned. Otherwise, an UnresolvedLookupExpr,
/// UnresolvedMemberExpr, DeclRefExpr, or MemberExpr will be returned.
/// The resulting expression is intended to be passed as the 'IdExpr'
/// argument to BuildSYCLReifyKernelParamsCallStmt() after the function
/// body has been parsed.
///
/// 'FD' must be a function declared with a valid sycl_kernel_entry_point
/// attribute and the current Sema context must match 'FD' so that scopes
/// are properly in place such that lookup is performed as if from the first
/// statement of the function body.
///
/// 'KIT' is the SYCL kernel info type to be used for lookup and it may
/// be a dependent type.
ExprResult BuildSYCLReifyKernelParamsIdExpr(Sema &SemaRef, FunctionDecl *FD,
                                            QualType KIT) {
  // The current context must be the function definition context to ensure
  // that name lookup is performed within the correct scope.
  assert(SemaRef.CurContext == FD && "The current declaration context does not "
                                     "match the requested function context");

  // An appropriate source location is required to emit diagnostics if
  // lookup fails to produce an overload set. The desired location is the
  // start of the function body, but that is not yet available since the
  // body of the function has not yet been set when this function is called.
  // The general location of the function is used instead.
  SourceLocation Loc = FD->getLocation();

  ASTContext &Ctx = SemaRef.getASTContext();
  IdentifierInfo &SYCLReifyKernelParamsID = Ctx.Idents.get(
      "sycl_reify_kernel_parameters", tok::TokenKind::identifier);

  // Establish a code synthesis context for the implicit name lookup of
  // a template named 'sycl_reify_kernel_parameters'. In the event of an error,
  // this ensures an appropriate diagnostic note is issued to explain why the
  // lookup was performed.
  Sema::CodeSynthesisContext CSC;
  CSC.Kind = Sema::CodeSynthesisContext::SYCLReifyKernelParamsLookup;
  CSC.Entity = FD;
  Sema::ScopedCodeSynthesisContext ScopedCSC(SemaRef, CSC);

  // Perform ordinary name lookup for a function or variable template that
  // accepts a single type template argument.
  LookupResult Result(SemaRef, &SYCLReifyKernelParamsID, Loc,
                      Sema::LookupOrdinaryName);
  CXXScopeSpec EmptySS;
  if (SemaRef.LookupTemplateName(Result, SemaRef.getCurScope(), EmptySS,
                                 /*ObjectType*/ QualType(),
                                 /*EnteringContext*/ false,
                                 Sema::TemplateNameIsRequired))
    return ExprError();
  if (Result.isAmbiguous())
    return ExprError();

  TemplateArgumentListInfo TALI{Loc, Loc};
  TemplateArgument KITA = TemplateArgument(KIT);
  TemplateArgumentLoc TAL =
      SemaRef.getTrivialTemplateArgumentLoc(KITA, QualType(), Loc);
  TALI.addArgument(TAL);

  ExprResult IdExpr;
  if (SemaRef.isPotentialImplicitMemberAccess(EmptySS, Result,
                                              /*IsAddressOfOperand*/ false)) {
    // The lookup result allows for a possible implicit member access that
    // would require an implicit or explicit 'this' argument.
    IdExpr = SemaRef.BuildPossibleImplicitMemberExpr(
        EmptySS, SourceLocation(), Result, &TALI, SemaRef.getCurScope());
  } else {
    IdExpr = SemaRef.BuildTemplateIdExpr(EmptySS, SourceLocation(), Result,
                                         /*RequiresADL*/ true, &TALI);
  }

  // The resulting expression may be invalid if, for example, 'FD' is a
  // non-static member function and name lookup selects a member function
  // (which would require a 'this' argument which is not available).
  if (IdExpr.isInvalid())
    return ExprError();

  return IdExpr;
}

// Constructs the arguments to be passed for the SYCL kernel launch call.
// Returns true on error and false otherwise.
bool BuildSYCLKernelParamsForwardingArgs(Sema &SemaRef, FunctionDecl *FD,
                                         SourceLocation Loc,
                                         SmallVectorImpl<Expr *> &Args,
                                         SmallVectorImpl<Expr *> &SRKPExprs) {
  // The current context must be the function definition context to ensure
  // that parameter references occur within the correct scope.
  assert(SemaRef.CurContext == FD && "The current declaration context does not "
                                     "match the requested function context");

  // Forward all parameters of 'FD' to the SYCL kernel launch function as if
  // by std::move().
  for (ParmVarDecl *PVD : FD->parameters()) {
    QualType ParamType = PVD->getOriginalType().getNonReferenceType();
    ExprResult E = SemaRef.BuildDeclRefExpr(PVD, ParamType, VK_LValue, Loc);
    if (E.isInvalid())
      return true;
    if (SRKPExprs.empty()) {
      // If no kernel parameters require reification, then by-value parameters
      // can be forwarded as xvalues. Otherwise, lvalues are required so that
      // they can be captured by reference by the callback lambda which will
      // then be called with lvalues referencing subobjects that require
      // reification.
      if (!PVD->getType()->isLValueReferenceType())
        E = ImplicitCastExpr::Create(SemaRef.Context, E.get()->getType(),
                                     CK_NoOp, E.get(), nullptr, VK_XValue,
                                     FPOptionsOverride());
      if (E.isInvalid())
        return true;
    }
    Args.push_back(E.get());
  }

  return false;
}

// Constructs the SYCL kernel launch call.
StmtResult BuildSYCLKernelLaunchCallStmt(Sema &SemaRef, FunctionDecl *FD,
                                         SourceLocation Loc, Expr *IdExpr,
                                         SmallVectorImpl<Expr *> &SRKPExprs) {
  SmallVector<Stmt *> Stmts;
  // IdExpr may be null if name lookup failed.
  if (IdExpr) {
    SmallVector<Expr *, 12> Args;

    // Establish a code synthesis context for construction of the arguments
    // for the implicit call to 'sycl_kernel_launch'.
    {
      Sema::CodeSynthesisContext CSC;
      CSC.Kind = Sema::CodeSynthesisContext::SYCLKernelLaunchLookup;
      CSC.Entity = FD;
      Sema::ScopedCodeSynthesisContext ScopedCSC(SemaRef, CSC);

      if (BuildSYCLKernelParamsForwardingArgs(SemaRef, FD, Loc, Args,
                                              SRKPExprs))
        return StmtError();
    }

    // Establish a code synthesis context for the implicit call to
    // 'sycl_kernel_launch'.
    ExprResult LaunchResult;
    {
      Sema::CodeSynthesisContext CSC;
      CSC.Kind = Sema::CodeSynthesisContext::SYCLKernelLaunchOverloadResolution;
      CSC.Entity = FD;
      CSC.CallArgs = Args.data();
      CSC.NumCallArgs = Args.size();
      Sema::ScopedCodeSynthesisContext ScopedCSC(SemaRef, CSC);

      LaunchResult =
          SemaRef.BuildCallExpr(SemaRef.getCurScope(), IdExpr, Loc, Args, Loc);
      if (LaunchResult.isInvalid())
        return StmtError();
    }

    // If kernel parameter reification is required or if 'sycl_kernel_launch'
    // doesn't return void, establish a code synthesis context for the implicit
    // call to the returned callable.
    if (!SRKPExprs.empty() || !LaunchResult.get()->getType()->isVoidType()) {
      Sema::CodeSynthesisContext CSC;
      CSC.Kind = Sema::CodeSynthesisContext::
          SYCLKernelLaunchCallbackOverloadResolution;
      CSC.Entity = FD;
      CSC.CallArgs = SRKPExprs.data();
      CSC.NumCallArgs = SRKPExprs.size();
      Sema::ScopedCodeSynthesisContext ScopedCSC(SemaRef, CSC);

      LaunchResult = SemaRef.BuildCallExpr(
          SemaRef.getCurScope(), LaunchResult.get(), Loc, SRKPExprs, Loc);
      if (LaunchResult.isInvalid())
        return StmtError();
    }

    Stmts.push_back(SemaRef.MaybeCreateExprWithCleanups(LaunchResult).get());
  }

  return CompoundStmt::Create(SemaRef.getASTContext(), Stmts,
                              FPOptionsOverride(), Loc, Loc);
}

// Constructs the SYCL reify kernel parameters call.
StmtResult
BuildSYCLReifyKernelParamsCallStmt(Sema &SemaRef, FunctionDecl *FD,
                                   SourceLocation Loc, Expr *IdExpr,
                                   SmallVectorImpl<Expr *> &SRKPExprs) {
  SmallVector<Stmt *> Stmts;
  // IdExpr may be null if name lookup failed.
  if (IdExpr) {
    SmallVector<Expr *, 12> Args;

    // Establish a code synthesis context for construction of the arguments
    // for the implicit call to 'sycl_reify_kernel_parameters'.
    {
      Sema::CodeSynthesisContext CSC;
      CSC.Kind = Sema::CodeSynthesisContext::SYCLReifyKernelParamsLookup;
      CSC.Entity = FD;
      Sema::ScopedCodeSynthesisContext ScopedCSC(SemaRef, CSC);

      if (BuildSYCLKernelParamsForwardingArgs(SemaRef, FD, Loc, Args,
                                              SRKPExprs))
        return StmtError();
    }

    // Establish a code synthesis context for the implicit call to
    // 'sycl_reify_kernel_parameters'.
    ExprResult LaunchResult;
    {
      Sema::CodeSynthesisContext CSC;
      CSC.Kind =
          Sema::CodeSynthesisContext::SYCLReifyKernelParamsOverloadResolution;
      CSC.Entity = FD;
      CSC.CallArgs = Args.data();
      CSC.NumCallArgs = Args.size();
      Sema::ScopedCodeSynthesisContext ScopedCSC(SemaRef, CSC);

      LaunchResult =
          SemaRef.BuildCallExpr(SemaRef.getCurScope(), IdExpr, Loc, Args, Loc);
      if (LaunchResult.isInvalid())
        return StmtError();
    }

    // If kernel parameter reification is required or if 'sycl_kernel_launch'
    // doesn't return void, establish a code synthesis context for the implicit
    // call to the returned callable.
    if (!SRKPExprs.empty() || !LaunchResult.get()->getType()->isVoidType()) {
      Sema::CodeSynthesisContext CSC;
      CSC.Kind = Sema::CodeSynthesisContext::
          SYCLReifyKernelParamsCallbackOverloadResolution;
      CSC.Entity = FD;
      CSC.CallArgs = SRKPExprs.data();
      CSC.NumCallArgs = SRKPExprs.size();
      Sema::ScopedCodeSynthesisContext ScopedCSC(SemaRef, CSC);

      LaunchResult = SemaRef.BuildCallExpr(
          SemaRef.getCurScope(), LaunchResult.get(), Loc, SRKPExprs, Loc);
      if (LaunchResult.isInvalid())
        return StmtError();
    }

    Stmts.push_back(SemaRef.MaybeCreateExprWithCleanups(LaunchResult).get());
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
        assert(SemaRef.getASTContext().hasSameUnqualifiedType(
            PVD->getType().getNonReferenceType(), VD->getType()));
        assert(!VD->getType().isMoreQualifiedThan(
            PVD->getType().getNonReferenceType(), SemaRef.getASTContext()));
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

DeclResult BuildSYCLKernelEntryPointOutline(Sema &SemaRef, FunctionDecl *FD,
                                            SourceLocation Loc,
                                            Expr *ReifyKernelParamsIdExpr,
                                            SmallVectorImpl<Expr *> &SRKPExprs,
                                            CompoundStmt *Body) {
  // If any of the kernel parameter types have a subobject type that requires
  // reification, then a call to sycl_reify_kernel_parameters() is synthesized
  // to provide the SYCL RT an opportunity to update them.
  if (!SRKPExprs.empty()) {
    StmtResult ReifyParamsResult = BuildSYCLReifyKernelParamsCallStmt(
        SemaRef, FD, Loc, ReifyKernelParamsIdExpr, SRKPExprs);
    if (ReifyParamsResult.isInvalid())
      return DeclResult(true);

    SmallVector<Stmt *> Stmts;
    Stmts.push_back(ReifyParamsResult.get());
    Stmts.push_back(Body);
    Body = CompoundStmt::Create(SemaRef.getASTContext(), Stmts,
                                FPOptionsOverride(), Loc, Loc);
  }

  OutlinedFunctionDecl *OFD = OutlinedFunctionDecl::Create(
      SemaRef.getASTContext(), FD, FD->getNumParams());

  // Create a map from the function parameters to the replacement set of
  // implicit parameters that will substitute for them in the outlined function
  // declaration.
  OutlinedFunctionDeclBodyInstantiator::ParmDeclMap ParmMap;
  unsigned i = 0;
  for (ParmVarDecl *PVD : FD->parameters()) {
    ImplicitParamDecl *IPD = ImplicitParamDecl::Create(
        SemaRef.getASTContext(), OFD, SourceLocation(), PVD->getIdentifier(),
        PVD->getType().getNonReferenceType(), ImplicitParamKind::Other);
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

bool isReifiedKernelParameterType(QualType Ty) {
  if (const auto *RT = Ty->getAsRecordDecl())
    return RT->getMostRecentDecl()->hasAttr<SYCLReifiedKernelParameterAttr>();
  return false;
}

class KernelParamsChecker : public ConstSubobjectVisitor<KernelParamsChecker> {
  SemaSYCL &SemaSYCLRef;
  SourceLocation Loc;
  using ObjectAccess =
      llvm::PointerUnion<const ParmVarDecl *, const CXXBaseSpecifier *,
                         const FieldDecl *>;
  SmallVector<ObjectAccess, 4> ObjectAccessPath;
  SmallVectorImpl<Expr *> &SRKPExprs;
  bool IsValid = true;

  void emitObjectAccessPathNotes() {
    for (auto Parent : llvm::reverse(ObjectAccessPath)) {
      if (auto *FD = dyn_cast<const FieldDecl *>(Parent)) {
        const CXXRecordDecl *ParentRD = cast<CXXRecordDecl>(FD->getParent());
        if (ParentRD->isLambda()) {
          SemaSYCLRef.Diag(ParentRD->getLocation(), diag::note_within_capture)
              << ParentRD->getCapture(FD->getFieldIndex())->getCapturedVar();
        } else {
          SemaSYCLRef.Diag(ParentRD->getLocation(),
                           diag::note_within_field_of_type)
              << ParentRD;
        }
      } else if (auto *BS = dyn_cast<const CXXBaseSpecifier *>(Parent)) {
        CXXRecordDecl *RD = BS->getType()->getAsCXXRecordDecl();
        assert(RD);
        SemaSYCLRef.Diag(BS->getBeginLoc(), diag::note_within_base_of_type)
            << RD;
      } else {
        auto *Param = cast<const ParmVarDecl *>(Parent);
        SemaSYCLRef.Diag(Param->getBeginLoc(), diag::note_within_param_of_type)
            << Param << Param->getType();
      }
    }
  }

  Expr *BuildParmVarDeclAccessExpr(const ParmVarDecl *PVD) {
    QualType Ty = PVD->getType().getNonReferenceType();
    return SemaSYCLRef.SemaRef.BuildDeclRefExpr(const_cast<ParmVarDecl *>(PVD),
                                                Ty, VK_LValue, Loc);
  }

  Expr *BuildDerivedToBaseCastExpr(Expr *E, const CXXBaseSpecifier *BS) {
    QualType DerivedTy = E->getType();
    QualType BaseTy = BS->getType();
    CXXCastPath BasePath;
    SemaSYCLRef.SemaRef.CheckDerivedToBaseConversion(DerivedTy, BaseTy, Loc,
                                                     SourceRange(), &BasePath,
                                                     /*IgnoreBaseAccess*/ true);
    auto *ICE = ImplicitCastExpr::Create(
        SemaSYCLRef.getASTContext(), BaseTy, CK_DerivedToBase, E,
        /* CXXCastPath=*/&BasePath, VK_LValue, FPOptionsOverride());
    return ICE;
  }

  Expr *BuildFieldDeclAccessExpr(Expr *E, const FieldDecl *FD) {
    DeclAccessPair DAP =
        DeclAccessPair::make(const_cast<FieldDecl *>(FD), AS_none);
    MemberExpr *ME = SemaSYCLRef.SemaRef.BuildMemberExpr(
        E, /*IsArrow */ false, Loc, NestedNameSpecifierLoc(), Loc,
        const_cast<FieldDecl *>(FD), DAP, /*HadMultipleCandidates*/ false,
        DeclarationNameInfo(FD->getDeclName(), Loc), FD->getType(), VK_LValue,
        OK_Ordinary);
    return ME;
  }

  void BuildSRKPObjectAccessExpr() {
    Expr *E = nullptr;
    for (auto OA : ObjectAccessPath) {
      if (auto *PVD = OA.dyn_cast<const ParmVarDecl *>()) {
        assert(!E && "No base expression should exist yet");
        E = BuildParmVarDeclAccessExpr(PVD);
      } else if (auto *BS = OA.dyn_cast<const CXXBaseSpecifier *>()) {
        assert(E && "A base expression should already exist");
        E = BuildDerivedToBaseCastExpr(E, BS);
      } else if (auto *FD = OA.dyn_cast<const FieldDecl *>()) {
        assert(E && "A base expression should already exist");
        E = BuildFieldDeclAccessExpr(E, FD);
      } else {
        llvm_unreachable("Unhandled object access kind");
      }
    }
    SRKPExprs.push_back(E);
  }

public:
  KernelParamsChecker(SemaSYCL &SR, SourceLocation Loc,
                      SmallVectorImpl<Expr *> &SRKPExprs)
      : ConstSubobjectVisitor<KernelParamsChecker>(SR.getASTContext()),
        SemaSYCLRef(SR), Loc(Loc), SRKPExprs(SRKPExprs) {}

  void checkParameter(const ParmVarDecl *PVD) {
    ObjectAccessPath.push_back(PVD);
    // Check the immediate type of the parameter.
    if (checkType(PVD->getType())) {
      // If type checking wasn't short circuited, visit subobjects to check
      // them.
      visit(PVD->getType());
    }
    ObjectAccessPath.pop_back();
    assert(ObjectAccessPath.empty());
  }

  bool visitBaseSpecifierPre(const CXXBaseSpecifier *BS) {
    ObjectAccessPath.push_back(BS);
    return checkType(BS->getType());
  }

  bool visitFieldDeclPre(const FieldDecl *FD) {
    ObjectAccessPath.push_back(FD);
    return checkType(FD->getType());
  }

  // Returns true if subobjects should be visited and false otherwise.
  bool checkType(QualType Ty) {
    if (isReifiedKernelParameterType(Ty)) {
      BuildSRKPObjectAccessExpr();
      // Don't visit subobjects of SRKP types; the SYCL RT takes full
      // responsibility for them.
      return false;
    }

    if (const auto *RT = Ty->getAs<const ReferenceType>()) {
      auto DirectParent = ObjectAccessPath.back();
      if (isa<const ParmVarDecl *>(DirectParent)) {
        // If the reference is for a kernel parameter, treat it like a value
        // parameter type. Reference parameter types are replaced by value
        // parameter types when the kernel entry point is generated.
        visit(RT->getPointeeType());
        return false;
      }

      assert(isa<const FieldDecl *>(DirectParent));
      auto *DirectFieldParent = cast<const FieldDecl *>(DirectParent);
      SemaSYCLRef.Diag(DirectFieldParent->getLocation(),
                       diag::err_bad_kernel_param_type)
          << DirectFieldParent->getType();
      emitObjectAccessPathNotes();

      // Don't visit the type of the reference since any further invalid
      // kernel parameter types contained within the referenced type
      // might not be relevant once the programmer addresses the
      // invalid use of a reference.
      IsValid = false;
      return false;
    }

    return true;
  }

  void visitFieldDeclPost(const FieldDecl *FD) { ObjectAccessPath.pop_back(); }
  void visitBaseSpecifierPost(const CXXBaseSpecifier *BS) {
    ObjectAccessPath.pop_back();
  }

  bool isInvalid() { return !IsValid; }
};

/// BuildResolvedSYCLKernelCallStmt builds a SYCLKernelCallStmt to serve as
/// the body of 'FD'. If construction fails, a diagnostic is issued and an
/// error statement is returned.
///
/// 'FD' must be a function declared with a valid sycl_kernel_entry_point
/// attribute and must not have had its body assigned yet.
///
/// 'Body' is the parsed compound statement body of 'FD' to be wrapped by
/// the new SYCLKernelCallStmt.
///
/// 'ASTFragments' contains the AST fragments previously returned by a call
/// to BuildSYCLKernelCallStmtASTFragments() for 'FD'.
StmtResult BuildResolvedSYCLKernelCallStmt(
    Sema &SemaRef, FunctionDecl *FD, CompoundStmt *Body,
    const SemaSYCL::SYCLKernelCallStmtASTFragments &ASTFragments) {
  assert(!FD->isInvalidDecl());
  assert(!FD->isTemplated());
  assert(FD->hasPrototype());

  ASTContext &Ctx = SemaRef.getASTContext();

  const auto *SKEPAttr = FD->getAttr<SYCLKernelEntryPointAttr>();
  assert(SKEPAttr && "Missing sycl_kernel_entry_point attribute");
  assert(!SKEPAttr->isInvalidAttr() &&
         "sycl_kernel_entry_point attribute is invalid");

  // Ensure that the kernel name was previously registered and that the
  // stored declaration matches.
  const SYCLKernelInfo &SKI = Ctx.getSYCLKernelInfo(SKEPAttr->getKernelName());
  assert(declaresSameEntity(SKI.getKernelEntryPointDecl(), FD) &&
         "SYCL kernel name conflict");

  // Check the function parameters for invalid parameter types and types that
  // require special handling.
  SmallVector<Expr *, 12> SRKPObjectAccessExprs;
  KernelParamsChecker KAC(SemaRef.SYCL(), FD->getLocation(),
                          SRKPObjectAccessExprs);
  for (auto P : FD->parameters())
    KAC.checkParameter(P);
  if (KAC.isInvalid())
    return StmtError();

  // Inject an explicit specialization definition for a specialization
  // of __sycl_kernel_info parameterized by the kernel name type.
  BuildSYCLKernelInfoClassSpecializationDefn(SemaRef, SKI.getKernelNameType());

  // Build the host kernel launch statement. An appropriate source location
  // is required to emit diagnostics.
  SourceLocation Loc = Body->getLBracLoc();
  StmtResult LaunchResult = BuildSYCLKernelLaunchCallStmt(
      SemaRef, FD, Loc, ASTFragments.KernelLaunchIdExpr, SRKPObjectAccessExprs);
  if (LaunchResult.isInvalid())
    return StmtError();

  // Build the outline of the synthesized device entry point function.
  DeclResult OFD = BuildSYCLKernelEntryPointOutline(
      SemaRef, FD, Loc, ASTFragments.ReifyKernelParamsIdExpr,
      SRKPObjectAccessExprs, Body);
  if (OFD.isInvalid())
    return StmtError();

  Stmt *NewBody = new (Ctx) SYCLKernelCallStmt(
      Body, LaunchResult.get(), cast<OutlinedFunctionDecl>(OFD.get()));

  return NewBody;
}

/// BuildUnresolvedSYCLKernelCallStmt constructs a UnresolvedSYCLKernelCallStmt
/// to serve as the body of a function in a dependent context.
///
/// 'Body' is the parsed compound statement body to be wrapped by the new
/// UnresolvedSYCLKernelCallStmt.
///
/// 'ASTFragments' contains the AST fragments previously returned by a call
/// to BuildSYCLKernelCallStmtASTFragments() for the function.
StmtResult BuildUnresolvedSYCLKernelCallStmt(
    Sema &SemaRef, CompoundStmt *Body,
    const SemaSYCL::SYCLKernelCallStmtASTFragments &ASTFragments) {
  return UnresolvedSYCLKernelCallStmt::Create(
      SemaRef.getASTContext(), Body, ASTFragments.KernelLaunchIdExpr,
      ASTFragments.ReifyKernelParamsIdExpr);
}

} // unnamed namespace

std::optional<SemaSYCL::SYCLKernelCallStmtASTFragments>
SemaSYCL::BuildSYCLKernelCallStmtASTFragments(FunctionDecl *FD) {
  const auto *SKEPAttr = FD->getAttr<SYCLKernelEntryPointAttr>();
  assert(SKEPAttr && "FD doesn't have a sycl_kernel_entry_point attribute");
  assert(!SKEPAttr->isInvalidAttr() &&
         "FD has an invalid sycl_kernel_entry_point attribute");
  QualType KernelName = SKEPAttr->getKernelName();

  // Retrieve or build the kernel info class template specialization type
  // needed for lookup of SYCL runtime declared templates.
  QualType KernelInfoType =
      GetSYCLKernelInfoClassSpecializationType(KernelName);
  assert(!KernelInfoType.isNull());

  // Perform lookup for the sycl_kernel_launch template.
  ExprResult KernelLaunchIdER =
      BuildSYCLKernelLaunchIdExpr(SemaRef, FD, KernelInfoType);
  if (KernelLaunchIdER.isInvalid())
    return {};

  // Perform lookup for the sycl_reify_kernel_parameters template. This
  // lookup is not required to succeed.
  ExprResult ReifyKernelParamsIdER =
      BuildSYCLReifyKernelParamsIdExpr(SemaRef, FD, KernelInfoType);

  return SemaSYCL::SYCLKernelCallStmtASTFragments{KernelLaunchIdER.get(),
                                                  ReifyKernelParamsIdER.get()};
}

StmtResult SemaSYCL::BuildSYCLKernelCallStmt(
    FunctionDecl *FD, CompoundStmt *Body,
    const SemaSYCL::SYCLKernelCallStmtASTFragments &ASTFragments) {
  // The current context must be the function definition context to ensure
  // that name lookup and parameter and local variable creation are performed
  // within the correct scope.
  assert(SemaRef.CurContext == FD && "The current declaration context does not "
                                     "match the requested function context");

  if (FD->isTemplated())
    return BuildUnresolvedSYCLKernelCallStmt(SemaRef, Body, ASTFragments);
  else
    return BuildResolvedSYCLKernelCallStmt(SemaRef, FD, Body, ASTFragments);
}
