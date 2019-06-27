//===--- SemaAPINotes.cpp - API Notes Handling ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the mapping from API notes to declaration attributes.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/AST/DeclObjC.h"
#include "clang/APINotes/APINotesReader.h"
using namespace clang;

namespace {
  enum IsActive_t : bool {
    IsNotActive,
    IsActive
  };
  enum IsReplacement_t : bool {
    IsNotReplacement,
    IsReplacement
  };

  struct VersionedInfoMetadata {
    /// An empty version refers to unversioned metadata.
    VersionTuple Version;
    unsigned IsActive: 1;
    unsigned IsReplacement: 1;

    VersionedInfoMetadata(VersionTuple version, IsActive_t active,
                          IsReplacement_t replacement)
      : Version(version), IsActive(active == IsActive_t::IsActive),
        IsReplacement(replacement == IsReplacement_t::IsReplacement) {}
  };
} // end anonymous namespace

/// Determine whether this is a multi-level pointer type.
static bool isMultiLevelPointerType(QualType type) {
  QualType pointee = type->getPointeeType();
  if (pointee.isNull())
    return false;

  return pointee->isAnyPointerType() || pointee->isObjCObjectPointerType() ||
         pointee->isMemberPointerType();
}

// Apply nullability to the given declaration.
static void applyNullability(Sema &S, Decl *decl, NullabilityKind nullability,
                             VersionedInfoMetadata metadata) {
  if (!metadata.IsActive)
    return;

  QualType type;

  // Nullability for a function/method appertains to the retain type.
  if (auto function = dyn_cast<FunctionDecl>(decl)) {
    type = function->getReturnType();
  } else if (auto method = dyn_cast<ObjCMethodDecl>(decl)) {
    type = method->getReturnType();
  } else if (auto value = dyn_cast<ValueDecl>(decl)) {
    type = value->getType();
  } else if (auto property = dyn_cast<ObjCPropertyDecl>(decl)) {
    type = property->getType();
  } else {
    return;
  }

  // Check the nullability specifier on this type.
  QualType origType = type;
  S.checkImplicitNullabilityTypeSpecifier(type, nullability,
                                          decl->getLocation(),
                                          isa<ParmVarDecl>(decl),
                                          /*overrideExisting=*/true);
  if (type.getTypePtr() == origType.getTypePtr())
    return;

  if (auto function = dyn_cast<FunctionDecl>(decl)) {
    const FunctionType *fnType = function->getType()->castAs<FunctionType>();
    if (const FunctionProtoType *proto = dyn_cast<FunctionProtoType>(fnType)) {
      function->setType(S.Context.getFunctionType(type, proto->getParamTypes(),
                                                  proto->getExtProtoInfo()));
    } else {
      function->setType(S.Context.getFunctionNoProtoType(type,
                                                         fnType->getExtInfo()));
    }
  } else if (auto method = dyn_cast<ObjCMethodDecl>(decl)) {
    method->setReturnType(type);

    // Make it a context-sensitive keyword if we can.
    if (!isMultiLevelPointerType(type)) {
      method->setObjCDeclQualifier(
        Decl::ObjCDeclQualifier(method->getObjCDeclQualifier() |
                                Decl::OBJC_TQ_CSNullability));
    }
  } else if (auto value = dyn_cast<ValueDecl>(decl)) {
    value->setType(type);

    // Make it a context-sensitive keyword if we can.
    if (auto parm = dyn_cast<ParmVarDecl>(decl)) {
      if (parm->isObjCMethodParameter() && !isMultiLevelPointerType(type)) {
        parm->setObjCDeclQualifier(
          Decl::ObjCDeclQualifier(parm->getObjCDeclQualifier() |
                                  Decl::OBJC_TQ_CSNullability));
      }
    }
  } else if (auto property = dyn_cast<ObjCPropertyDecl>(decl)) {
    property->setType(type, property->getTypeSourceInfo());

    // Make it a property attribute if we can.
    if (!isMultiLevelPointerType(type)) {
      property->setPropertyAttributes(
        ObjCPropertyDecl::OBJC_PR_null_resettable);
    }
  } else {
    llvm_unreachable("cannot handle nullability here");
  }
}

/// Copy a string into ASTContext-allocated memory.
static StringRef CopyString(ASTContext &ctx, StringRef string) {
  void *mem = ctx.Allocate(string.size(), alignof(char));
  memcpy(mem, string.data(), string.size());
  return StringRef(static_cast<char *>(mem), string.size());
}

namespace {
  template <typename A>
  struct AttrKindFor {};

#define ATTR(X) \
  template <> struct AttrKindFor<X##Attr> { \
    static const attr::Kind value = attr::X; \
  };
#include "clang/Basic/AttrList.inc"

  /// Handle an attribute introduced by API notes.
  ///
  /// \param shouldAddAttribute Whether we should add a new attribute
  /// (otherwise, we might remove an existing attribute).
  /// \param createAttr Create the new attribute to be added.
  template<typename A>
  void handleAPINotedAttribute(
         Sema &S, Decl *D, bool shouldAddAttribute,
         VersionedInfoMetadata metadata,
         llvm::function_ref<A *()> createAttr,
         llvm::function_ref<Decl::attr_iterator(const Decl*)> getExistingAttr) {
    if (metadata.IsActive) {
      auto existing = getExistingAttr(D);
      if (existing != D->attr_end()) {
        // Remove the existing attribute, and treat it as a superseded
        // non-versioned attribute.
        auto *versioned = SwiftVersionedAttr::CreateImplicit(
            S.Context, metadata.Version, *existing, /*IsReplacedByActive*/true);

        D->getAttrs().erase(existing);
        D->addAttr(versioned);
      }

      // If we're supposed to add a new attribute, do so.
      if (shouldAddAttribute) {
        if (auto attr = createAttr()) {
          D->addAttr(attr);
        }
      }

    } else {
      if (shouldAddAttribute) {
        if (auto attr = createAttr()) {
          auto *versioned = SwiftVersionedAttr::CreateImplicit(
              S.Context, metadata.Version, attr,
              /*IsReplacedByActive*/metadata.IsReplacement);
          D->addAttr(versioned);
        }
      } else {
        // FIXME: This isn't preserving enough information for things like
        // availability, where we're trying to remove a /specific/ kind of
        // attribute.
        auto *versioned = SwiftVersionedRemovalAttr::CreateImplicit(
            S.Context,  metadata.Version, AttrKindFor<A>::value,
            /*IsReplacedByActive*/metadata.IsReplacement);
        D->addAttr(versioned);
      }
    }
  }
  
  template<typename A>
  void handleAPINotedAttribute(
         Sema &S, Decl *D, bool shouldAddAttribute,
         VersionedInfoMetadata metadata,
         llvm::function_ref<A *()> createAttr) {
    handleAPINotedAttribute<A>(S, D, shouldAddAttribute, metadata, createAttr,
                               [](const Decl *decl) {
      return llvm::find_if(decl->attrs(), [](const Attr *next) {
        return isa<A>(next);
      });
    });
  }
}

template <typename A = CFReturnsRetainedAttr>
static void handleAPINotedRetainCountAttribute(Sema &S, Decl *D,
                                               bool shouldAddAttribute,
                                               VersionedInfoMetadata metadata) {
  // The template argument has a default to make the "removal" case more
  // concise; it doesn't matter /which/ attribute is being removed.
  handleAPINotedAttribute<A>(S, D, shouldAddAttribute, metadata, [&] {
    return new (S.Context) A(SourceRange(), S.Context, /*SpellingIndex*/0);
  }, [](const Decl *D) -> Decl::attr_iterator {
    return llvm::find_if(D->attrs(), [](const Attr *next) -> bool {
      return isa<CFReturnsRetainedAttr>(next) ||
             isa<CFReturnsNotRetainedAttr>(next) ||
             isa<NSReturnsRetainedAttr>(next) ||
             isa<NSReturnsNotRetainedAttr>(next) ||
             isa<CFAuditedTransferAttr>(next);
    });
  });
}

static void handleAPINotedRetainCountConvention(
    Sema &S, Decl *D, VersionedInfoMetadata metadata,
    Optional<api_notes::RetainCountConventionKind> convention) {
  if (!convention)
    return;
  switch (convention.getValue()) {
  case api_notes::RetainCountConventionKind::None:
    if (isa<FunctionDecl>(D)) {
      handleAPINotedRetainCountAttribute<CFUnknownTransferAttr>(
          S, D, /*shouldAddAttribute*/true, metadata);
    } else {
      handleAPINotedRetainCountAttribute(S, D, /*shouldAddAttribute*/false,
                                         metadata);
    }
    break;
  case api_notes::RetainCountConventionKind::CFReturnsRetained:
    handleAPINotedRetainCountAttribute<CFReturnsRetainedAttr>(
        S, D, /*shouldAddAttribute*/true, metadata);
    break;
  case api_notes::RetainCountConventionKind::CFReturnsNotRetained:
    handleAPINotedRetainCountAttribute<CFReturnsNotRetainedAttr>(
        S, D, /*shouldAddAttribute*/true, metadata);
    break;
  case api_notes::RetainCountConventionKind::NSReturnsRetained:
    handleAPINotedRetainCountAttribute<NSReturnsRetainedAttr>(
        S, D, /*shouldAddAttribute*/true, metadata);
    break;
  case api_notes::RetainCountConventionKind::NSReturnsNotRetained:
    handleAPINotedRetainCountAttribute<NSReturnsNotRetainedAttr>(
        S, D, /*shouldAddAttribute*/true, metadata);
    break;
  }
}

static void ProcessAPINotes(Sema &S, Decl *D,
                            const api_notes::CommonEntityInfo &info,
                            VersionedInfoMetadata metadata) {
  // Availability
  if (info.Unavailable) {
    handleAPINotedAttribute<UnavailableAttr>(S, D, true, metadata,
      [&] {
        return new (S.Context) UnavailableAttr(SourceRange(), S.Context,
                                               CopyString(S.Context,
                                                          info.UnavailableMsg),
                                               /*SpellingIndex*/0);
    });
  }

  if (info.UnavailableInSwift) {
    handleAPINotedAttribute<AvailabilityAttr>(S, D, true, metadata, [&] {
      return new (S.Context) AvailabilityAttr(SourceRange(), S.Context,
                                              &S.Context.Idents.get("swift"),
                                              VersionTuple(), VersionTuple(),
                                              VersionTuple(),
                                              /*Unavailable=*/true,
                                              CopyString(S.Context,
                                                         info.UnavailableMsg),
                                              /*Strict=*/false,
                                              /*Replacement=*/StringRef(),
                                              /*Priority=*/Sema::AP_Explicit,
                                              /*SpellingIndex*/0);
    },
    [](const Decl *decl) {
      return llvm::find_if(decl->attrs(), [](const Attr *next) -> bool {
        auto *AA = dyn_cast<AvailabilityAttr>(next);
        if (!AA)
          return false;
        const IdentifierInfo *platform = AA->getPlatform();
        if (!platform)
          return false;
        return platform->isStr("swift");
      });
    });
  }

  // swift_private
  if (auto swiftPrivate = info.isSwiftPrivate()) {
    handleAPINotedAttribute<SwiftPrivateAttr>(S, D, *swiftPrivate, metadata,
                                              [&] {
      return new (S.Context) SwiftPrivateAttr(SourceRange(), S.Context,
                                              /*SpellingIndex*/0);
    });
  }

  // swift_name
  if (!info.SwiftName.empty()) {
    handleAPINotedAttribute<SwiftNameAttr>(S, D, true, metadata,
                                           [&]() -> SwiftNameAttr * {
      auto &APINoteName = S.getASTContext().Idents.get("SwiftName API Note");
      
      if (!S.DiagnoseSwiftName(D, info.SwiftName, D->getLocation(),
                               &APINoteName)) {
        return nullptr;
      }

      return new (S.Context) SwiftNameAttr(SourceRange(), S.Context,
                                           CopyString(S.Context,
                                                      info.SwiftName),
                                           /*SpellingIndex*/0);
    });
  }
}

static void ProcessAPINotes(Sema &S, Decl *D,
                            const api_notes::CommonTypeInfo &info,
                            VersionedInfoMetadata metadata) {
  // swift_bridge
  if (auto swiftBridge = info.getSwiftBridge()) {
    handleAPINotedAttribute<SwiftBridgeAttr>(S, D, !swiftBridge->empty(),
                                             metadata, [&] {
      return new (S.Context) SwiftBridgeAttr(SourceRange(), S.Context,
                                             CopyString(S.Context,
                                                        *swiftBridge),
                                             /*SpellingIndex*/0);
    });
  }

  // ns_error_domain
  if (auto nsErrorDomain = info.getNSErrorDomain()) {
    handleAPINotedAttribute<NSErrorDomainAttr>(S, D, !nsErrorDomain->empty(),
                                               metadata, [&] {
      return new (S.Context) NSErrorDomainAttr(
          SourceRange(), S.Context, &S.Context.Idents.get(*nsErrorDomain),
          /*SpellingIndex*/0);
    });
  }

  ProcessAPINotes(S, D, static_cast<const api_notes::CommonEntityInfo &>(info),
                  metadata);
}

/// Check that the replacement type provided by API notes is reasonable.
///
/// This is a very weak form of ABI check.
static bool checkAPINotesReplacementType(Sema &S, SourceLocation loc,
                                         QualType origType,
                                         QualType replacementType) {
  if (S.Context.getTypeSize(origType) !=
        S.Context.getTypeSize(replacementType)) {
    S.Diag(loc, diag::err_incompatible_replacement_type)
      << replacementType << origType;
    return true;
  }

  return false;
}

/// Process API notes for a variable or property.
static void ProcessAPINotes(Sema &S, Decl *D,
                            const api_notes::VariableInfo &info,
                            VersionedInfoMetadata metadata) {
  // Type override.
  if (metadata.IsActive && !info.getType().empty() &&
      S.ParseTypeFromStringCallback) {
    auto parsedType = S.ParseTypeFromStringCallback(info.getType(),
                                                    "<API Notes>",
                                                    D->getLocation());
    if (parsedType.isUsable()) {
      QualType type = Sema::GetTypeFromParser(parsedType.get());
      auto typeInfo =
        S.Context.getTrivialTypeSourceInfo(type, D->getLocation());

      if (auto var = dyn_cast<VarDecl>(D)) {
        // Make adjustments to parameter types.
        if (isa<ParmVarDecl>(var)) {
          type = S.adjustParameterTypeForObjCAutoRefCount(type,
                                                          D->getLocation(),
                                                          typeInfo);
          type = S.Context.getAdjustedParameterType(type);
        }

        if (!checkAPINotesReplacementType(S, var->getLocation(), var->getType(),
                                          type)) {
          var->setType(type);
          var->setTypeSourceInfo(typeInfo);
        }
      } else if (auto property = dyn_cast<ObjCPropertyDecl>(D)) {
        if (!checkAPINotesReplacementType(S, property->getLocation(),
                                          property->getType(),
                                          type)) {
          property->setType(type, typeInfo);
        }
      } else {
        llvm_unreachable("API notes allowed a type on an unknown declaration");
      }
    }
  }

  // Nullability.
  if (auto Nullability = info.getNullability()) {
    applyNullability(S, D, *Nullability, metadata);
  }

  // Handle common entity information.
  ProcessAPINotes(S, D, static_cast<const api_notes::CommonEntityInfo &>(info),
                  metadata);
}

/// Process API notes for a parameter.
static void ProcessAPINotes(Sema &S, ParmVarDecl *D,
                            const api_notes::ParamInfo &info,
                            VersionedInfoMetadata metadata) {
  // noescape
  if (auto noescape = info.isNoEscape()) {
    handleAPINotedAttribute<NoEscapeAttr>(S, D, *noescape, metadata, [&] {
      return new (S.Context) NoEscapeAttr(SourceRange(), S.Context,
                                          /*SpellingIndex*/0);
    });
  }

  // Retain count convention
  handleAPINotedRetainCountConvention(S, D, metadata,
                                      info.getRetainCountConvention());

  // Handle common entity information.
  ProcessAPINotes(S, D, static_cast<const api_notes::VariableInfo &>(info),
                  metadata);
}

/// Process API notes for a global variable.
static void ProcessAPINotes(Sema &S, VarDecl *D,
                            const api_notes::GlobalVariableInfo &info,
                            VersionedInfoMetadata metadata) {
  // Handle common entity information.
  ProcessAPINotes(S, D, static_cast<const api_notes::VariableInfo &>(info),
                  metadata);
}

/// Process API notes for an Objective-C property.
static void ProcessAPINotes(Sema &S, ObjCPropertyDecl *D,
                            const api_notes::ObjCPropertyInfo &info,
                            VersionedInfoMetadata metadata) {
  // Handle common entity information.
  ProcessAPINotes(S, D, static_cast<const api_notes::VariableInfo &>(info),
                  metadata);
  if (auto asAccessors = info.getSwiftImportAsAccessors()) {
    handleAPINotedAttribute<SwiftImportPropertyAsAccessorsAttr>(S, D,
                                                                *asAccessors,
                                                                metadata, [&] {
      return new (S.Context) SwiftImportPropertyAsAccessorsAttr(
          SourceRange(), S.Context, /*SpellingIndex*/0);
    });
  }
}

namespace {
  typedef llvm::PointerUnion<FunctionDecl *, ObjCMethodDecl *> FunctionOrMethod;
}

/// Process API notes for a function or method.
static void ProcessAPINotes(Sema &S, FunctionOrMethod AnyFunc,
                            const api_notes::FunctionInfo &info,
                            VersionedInfoMetadata metadata) {
  // Find the declaration itself.
  FunctionDecl *FD = AnyFunc.dyn_cast<FunctionDecl *>();
  Decl *D = FD;
  ObjCMethodDecl *MD = 0;
  if (!D) {
    MD = AnyFunc.get<ObjCMethodDecl *>();
    D = MD;
  }

  // Nullability of return type.
  if (info.NullabilityAudited) {
    applyNullability(S, D, info.getReturnTypeInfo(), metadata);
  }

  // Parameters.
  unsigned NumParams;
  if (FD)
    NumParams = FD->getNumParams();
  else
    NumParams = MD->param_size();

  bool anyTypeChanged = false;
  for (unsigned I = 0; I != NumParams; ++I) {
    ParmVarDecl *Param;
    if (FD)
      Param = FD->getParamDecl(I);
    else
      Param = MD->param_begin()[I];

    QualType paramTypeBefore = Param->getType();

    if (I < info.Params.size()) {
      ProcessAPINotes(S, Param, info.Params[I], metadata);
    }

    // Nullability.
    if (info.NullabilityAudited)
      applyNullability(S, Param, info.getParamTypeInfo(I), metadata);

    if (paramTypeBefore.getAsOpaquePtr() != Param->getType().getAsOpaquePtr())
      anyTypeChanged = true;
  }

  // Result type override.
  QualType overriddenResultType;
  if (metadata.IsActive && !info.ResultType.empty() &&
      S.ParseTypeFromStringCallback) {
    auto parsedType = S.ParseTypeFromStringCallback(info.ResultType,
                                                    "<API Notes>",
                                                    D->getLocation());
    if (parsedType.isUsable()) {
      QualType resultType = Sema::GetTypeFromParser(parsedType.get());

      if (MD) {
        if (!checkAPINotesReplacementType(S, D->getLocation(),
                                          MD->getReturnType(), resultType)) {
          auto resultTypeInfo =
            S.Context.getTrivialTypeSourceInfo(resultType, D->getLocation());
          MD->setReturnType(resultType);
          MD->setReturnTypeSourceInfo(resultTypeInfo);
        }
      } else if (!checkAPINotesReplacementType(S, FD->getLocation(),
                                               FD->getReturnType(),
                                               resultType)) {
        overriddenResultType = resultType;
        anyTypeChanged = true;
      }
    }
  }

  // If the result type or any of the parameter types changed for a function
  // declaration, we have to rebuild the type.
  if (FD && anyTypeChanged) {
    if (const auto *fnProtoType = FD->getType()->getAs<FunctionProtoType>()) {
      if (overriddenResultType.isNull())
        overriddenResultType = fnProtoType->getReturnType();

      SmallVector<QualType, 4> paramTypes;
      for (auto param : FD->parameters()) {
        paramTypes.push_back(param->getType());
      }
      FD->setType(S.Context.getFunctionType(overriddenResultType,
                                            paramTypes,
                                            fnProtoType->getExtProtoInfo()));
    } else if (!overriddenResultType.isNull()) {
      const auto *fnNoProtoType = FD->getType()->castAs<FunctionNoProtoType>();
      FD->setType(
              S.Context.getFunctionNoProtoType(overriddenResultType,
                                               fnNoProtoType->getExtInfo()));
    }
  }

  // Retain count convention
  handleAPINotedRetainCountConvention(S, D, metadata,
                                      info.getRetainCountConvention());

  // Handle common entity information.
  ProcessAPINotes(S, D, static_cast<const api_notes::CommonEntityInfo &>(info),
                  metadata);
}

/// Process API notes for a global function.
static void ProcessAPINotes(Sema &S, FunctionDecl *D,
                            const api_notes::GlobalFunctionInfo &info,
                            VersionedInfoMetadata metadata) {

  // Handle common function information.
  ProcessAPINotes(S, FunctionOrMethod(D),
                  static_cast<const api_notes::FunctionInfo &>(info), metadata);
}

/// Process API notes for an enumerator.
static void ProcessAPINotes(Sema &S, EnumConstantDecl *D,
                            const api_notes::EnumConstantInfo &info,
                            VersionedInfoMetadata metadata) {

  // Handle common information.
  ProcessAPINotes(S, D,
                  static_cast<const api_notes::CommonEntityInfo &>(info),
                  metadata);
}

/// Process API notes for an Objective-C method.
static void ProcessAPINotes(Sema &S, ObjCMethodDecl *D,
                            const api_notes::ObjCMethodInfo &info,
                            VersionedInfoMetadata metadata) {
  // Designated initializers.
  if (info.DesignatedInit) {
    handleAPINotedAttribute<ObjCDesignatedInitializerAttr>(S, D, true, metadata,
                                                           [&] {
      if (ObjCInterfaceDecl *IFace = D->getClassInterface()) {
        IFace->setHasDesignatedInitializers();
      }
      return new (S.Context) ObjCDesignatedInitializerAttr(SourceRange(),
                                                           S.Context,
                                                           /*SpellingIndex*/0);
    });
  }

  // Handle common function information.
  ProcessAPINotes(S, FunctionOrMethod(D),
                  static_cast<const api_notes::FunctionInfo &>(info), metadata);
}

/// Process API notes for a tag.
static void ProcessAPINotes(Sema &S, TagDecl *D,
                            const api_notes::TagInfo &info,
                            VersionedInfoMetadata metadata) {
  if (auto extensibility = info.EnumExtensibility) {
    using api_notes::EnumExtensibilityKind;
    bool shouldAddAttribute = (*extensibility != EnumExtensibilityKind::None);
    handleAPINotedAttribute<EnumExtensibilityAttr>(S, D, shouldAddAttribute,
                                                   metadata, [&] {
      EnumExtensibilityAttr::Kind kind;
      switch (extensibility.getValue()) {
      case EnumExtensibilityKind::None:
        llvm_unreachable("remove only");
      case EnumExtensibilityKind::Open:
        kind = EnumExtensibilityAttr::Open;
        break;
      case EnumExtensibilityKind::Closed:
        kind = EnumExtensibilityAttr::Closed;
        break;
      }
      return new (S.Context) EnumExtensibilityAttr(SourceRange(), S.Context,
                                                   kind, /*SpellingIndex*/0);
    });
  }

  if (auto flagEnum = info.isFlagEnum()) {
    handleAPINotedAttribute<FlagEnumAttr>(S, D, flagEnum.getValue(), metadata,
                                          [&] {
      return new (S.Context) FlagEnumAttr(SourceRange(), S.Context,
                                          /*SpellingIndex*/0);
    });
  }

  // Handle common type information.
  ProcessAPINotes(S, D, static_cast<const api_notes::CommonTypeInfo &>(info),
                  metadata);
}

/// Process API notes for a typedef.
static void ProcessAPINotes(Sema &S, TypedefNameDecl *D,
                            const api_notes::TypedefInfo &info,
                            VersionedInfoMetadata metadata) {
  // swift_wrapper
  using SwiftWrapperKind = api_notes::SwiftWrapperKind;

  if (auto swiftWrapper = info.SwiftWrapper) {
    handleAPINotedAttribute<SwiftNewtypeAttr>(S, D,
      *swiftWrapper != SwiftWrapperKind::None, metadata,
      [&] {
        SwiftNewtypeAttr::NewtypeKind kind;
        switch (*swiftWrapper) {
        case SwiftWrapperKind::None:
          llvm_unreachable("Shouldn't build an attribute");

        case SwiftWrapperKind::Struct:
          kind = SwiftNewtypeAttr::NK_Struct;
          break;

        case SwiftWrapperKind::Enum:
          kind = SwiftNewtypeAttr::NK_Enum;
          break;
        }
        return new (S.Context) SwiftNewtypeAttr(
            SourceRange(), S.Context, kind,
            SwiftNewtypeAttr::GNU_swift_wrapper);
    });
  }

  // Handle common type information.
  ProcessAPINotes(S, D, static_cast<const api_notes::CommonTypeInfo &>(info),
                  metadata);
}

/// Process API notes for an Objective-C class or protocol.
static void ProcessAPINotes(Sema &S, ObjCContainerDecl *D,
                            const api_notes::ObjCContextInfo &info,
                            VersionedInfoMetadata metadata) {

  // Handle common type information.
  ProcessAPINotes(S, D, static_cast<const api_notes::CommonTypeInfo &>(info),
                  metadata);
}

/// Process API notes for an Objective-C class.
static void ProcessAPINotes(Sema &S, ObjCInterfaceDecl *D,
                            const api_notes::ObjCContextInfo &info,
                            VersionedInfoMetadata metadata) {
  if (auto asNonGeneric = info.getSwiftImportAsNonGeneric()) {
    handleAPINotedAttribute<SwiftImportAsNonGenericAttr>(S, D, *asNonGeneric,
                                                         metadata, [&] {
      return new (S.Context) SwiftImportAsNonGenericAttr(SourceRange(),
                                                         S.Context,
                                                         /*SpellingIndex*/0);
    });
  }

    if (auto objcMembers = info.getSwiftObjCMembers()) {
    handleAPINotedAttribute<SwiftObjCMembersAttr>(S, D, *objcMembers,
                                                         metadata, [&] {
      return new (S.Context) SwiftObjCMembersAttr(SourceRange(), S.Context,
                                                  /*SpellingIndex*/0);
    });
  }

  // Handle information common to Objective-C classes and protocols.
  ProcessAPINotes(S, static_cast<clang::ObjCContainerDecl *>(D), info,
                  metadata);
}

/// If we're applying API notes with an active, non-default version, and the
/// versioned API notes have a SwiftName but the declaration normally wouldn't
/// have one, add a removal attribute to make it clear that the new SwiftName
/// attribute only applies to the active version of \p D, not to all versions.
///
/// This must be run \em before processing API notes for \p D, because otherwise
/// any existing SwiftName attribute will have been packaged up in a 
/// SwiftVersionedAttr.
template <typename SpecificInfo>
static void maybeAttachUnversionedSwiftName(
    Sema &S, Decl *D,
    const api_notes::APINotesReader::VersionedInfo<SpecificInfo> Info) {
  if (D->hasAttr<SwiftNameAttr>())
    return;
  if (!Info.getSelected())
    return;

  // Is the active slice versioned, and does it set a Swift name?
  VersionTuple SelectedVersion;
  SpecificInfo SelectedInfoSlice;
  std::tie(SelectedVersion, SelectedInfoSlice) = Info[*Info.getSelected()];
  if (SelectedVersion.empty())
    return;
  if (SelectedInfoSlice.SwiftName.empty())
    return;

  // Does the unversioned slice /not/ set a Swift name?
  for (const auto &VersionAndInfoSlice : Info) {
    if (!VersionAndInfoSlice.first.empty())
      continue;
    if (!VersionAndInfoSlice.second.SwiftName.empty())
      return;
  }

  // Then explicitly call that out with a removal attribute.
  VersionedInfoMetadata DummyFutureMetadata(SelectedVersion, IsNotActive,
                                            IsReplacement);
  handleAPINotedAttribute<SwiftNameAttr>(S, D, /*add*/false,
                                         DummyFutureMetadata,
                                         []() -> SwiftNameAttr * {
    llvm_unreachable("should not try to add an attribute here");
  });
}

/// Processes all versions of versioned API notes.
///
/// Just dispatches to the various ProcessAPINotes functions in this file.
template <typename SpecificDecl, typename SpecificInfo>
static void ProcessVersionedAPINotes(
    Sema &S, SpecificDecl *D,
    const api_notes::APINotesReader::VersionedInfo<SpecificInfo> Info) {

  maybeAttachUnversionedSwiftName(S, D, Info);

  unsigned Selected = Info.getSelected().getValueOr(Info.size());

  VersionTuple Version;
  SpecificInfo InfoSlice;
  for (unsigned i = 0, e = Info.size(); i != e; ++i) {
    std::tie(Version, InfoSlice) = Info[i];
    auto Active = (i == Selected) ? IsActive : IsNotActive;
    auto Replacement = IsNotReplacement;
    if (Active == IsNotActive && Version.empty()) {
      Replacement = IsReplacement;
      Version = Info[Selected].first;
    }
    ProcessAPINotes(S, D, InfoSlice, VersionedInfoMetadata(Version, Active,
                                                           Replacement));
  }
}

/// Process API notes that are associated with this declaration, mapping them
/// to attributes as appropriate.
void Sema::ProcessAPINotes(Decl *D) {
  if (!D)
    return;

  // Globals.
  if (D->getDeclContext()->isFileContext()) {
    // Global variables.
    if (auto VD = dyn_cast<VarDecl>(D)) {
      for (auto Reader : APINotes.findAPINotes(D->getLocation())) {
        auto Info = Reader->lookupGlobalVariable(VD->getName());
        ProcessVersionedAPINotes(*this, VD, Info);
      }

      return;
    }

    // Global functions.
    if (auto FD = dyn_cast<FunctionDecl>(D)) {
      if (FD->getDeclName().isIdentifier()) {
        for (auto Reader : APINotes.findAPINotes(D->getLocation())) {
          auto Info = Reader->lookupGlobalFunction(FD->getName());
          ProcessVersionedAPINotes(*this, FD, Info);
        }
      }

      return;
    }

    // Objective-C classes.
    if (auto Class = dyn_cast<ObjCInterfaceDecl>(D)) {
      for (auto Reader : APINotes.findAPINotes(D->getLocation())) {
        auto Info = Reader->lookupObjCClassInfo(Class->getName());
        ProcessVersionedAPINotes(*this, Class, Info);
      }

      return;
    }

    // Objective-C protocols.
    if (auto Protocol = dyn_cast<ObjCProtocolDecl>(D)) {
      for (auto Reader : APINotes.findAPINotes(D->getLocation())) {
        auto Info = Reader->lookupObjCProtocolInfo(Protocol->getName());
        ProcessVersionedAPINotes(*this, Protocol, Info);
      }

      return;
    }

    // Tags
    if (auto Tag = dyn_cast<TagDecl>(D)) {
      for (auto Reader : APINotes.findAPINotes(D->getLocation())) {
        auto Info = Reader->lookupTag(Tag->getName());
        ProcessVersionedAPINotes(*this, Tag, Info);
      }

      return;
    }

    // Typedefs
    if (auto Typedef = dyn_cast<TypedefNameDecl>(D)) {
      for (auto Reader : APINotes.findAPINotes(D->getLocation())) {
        auto Info = Reader->lookupTypedef(Typedef->getName());
        ProcessVersionedAPINotes(*this, Typedef, Info);
      }

      return;
    }

    return;
  }

  // Enumerators.
  if (D->getDeclContext()->getRedeclContext()->isFileContext()) {
    if (auto EnumConstant = dyn_cast<EnumConstantDecl>(D)) {
      for (auto Reader : APINotes.findAPINotes(D->getLocation())) {
        auto Info = Reader->lookupEnumConstant(EnumConstant->getName());
        ProcessVersionedAPINotes(*this, EnumConstant, Info);
      }

      return;
    }
  }

  if (auto ObjCContainer = dyn_cast<ObjCContainerDecl>(D->getDeclContext())) {
    // Location function that looks up an Objective-C context.
    auto GetContext = [&](api_notes::APINotesReader *Reader)
                        -> Optional<api_notes::ContextID> {
      if (auto Protocol = dyn_cast<ObjCProtocolDecl>(ObjCContainer)) {
        if (auto Found = Reader->lookupObjCProtocolID(Protocol->getName()))
          return *Found;

        return None;
      }

      if (auto Impl = dyn_cast<ObjCCategoryImplDecl>(ObjCContainer)) {
        if (auto Cat = Impl->getCategoryDecl())
          ObjCContainer = Cat;
        else
          return None;
      }

      if (auto Category = dyn_cast<ObjCCategoryDecl>(ObjCContainer)) {
        if (Category->getClassInterface())
          ObjCContainer = Category->getClassInterface();
        else
          return None;
      }

      if (auto Impl = dyn_cast<ObjCImplDecl>(ObjCContainer)) {
        if (Impl->getClassInterface())
          ObjCContainer = Impl->getClassInterface();
        else
          return None;
      }

      if (auto Class = dyn_cast<ObjCInterfaceDecl>(ObjCContainer)) {
        if (auto Found = Reader->lookupObjCClassID(Class->getName()))
          return *Found;

        return None;

      }

      return None;
    };

    // Objective-C methods.
    if (auto Method = dyn_cast<ObjCMethodDecl>(D)) {
      for (auto Reader : APINotes.findAPINotes(D->getLocation())) {
        if (auto Context = GetContext(Reader)) {
          // Map the selector.
          Selector Sel = Method->getSelector();
          SmallVector<StringRef, 2> SelPieces;
          if (Sel.isUnarySelector())
            SelPieces.push_back(Sel.getNameForSlot(0));
          else {
            for (unsigned i = 0, n = Sel.getNumArgs(); i != n; ++i)
              SelPieces.push_back(Sel.getNameForSlot(i));
          }

          api_notes::ObjCSelectorRef SelectorRef;
          SelectorRef.NumPieces = Sel.getNumArgs();
          SelectorRef.Identifiers = SelPieces;

          auto Info = Reader->lookupObjCMethod(*Context, SelectorRef,
                                               Method->isInstanceMethod());
          ProcessVersionedAPINotes(*this, Method, Info);
        }
      }
    }

    // Objective-C properties.
    if (auto Property = dyn_cast<ObjCPropertyDecl>(D)) {
      for (auto Reader : APINotes.findAPINotes(D->getLocation())) {
        if (auto Context = GetContext(Reader)) {
          bool isInstanceProperty =
            (Property->getPropertyAttributesAsWritten() &
               ObjCPropertyDecl::OBJC_PR_class) == 0;
          auto Info = Reader->lookupObjCProperty(*Context, Property->getName(),
                                                 isInstanceProperty);
          ProcessVersionedAPINotes(*this, Property, Info);
        }
      }

      return;
    }

    return;
  }
}
