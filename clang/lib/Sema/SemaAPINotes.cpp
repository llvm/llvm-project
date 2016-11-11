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
using clang::api_notes::VersionedInfoRole;

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
                             VersionedInfoRole role) {
  bool overrideExisting;
  switch (role) {
  case VersionedInfoRole::AugmentSource:
    overrideExisting = false;
    break;

  case VersionedInfoRole::ReplaceSource:
    overrideExisting = true;
    break;

  case VersionedInfoRole::Versioned:
    // FIXME: Record versioned info?
    return;
  }

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
  S.checkNullabilityTypeSpecifier(type, nullability, decl->getLocation(),
                                  /*isContextSensitive=*/false,
                                  isa<ParmVarDecl>(decl), /*implicit=*/true,
                                  overrideExisting);
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
  /// Handle an attribute introduced by API notes.
  ///
  /// \param shouldAddAttribute Whether we should add a new attribute
  /// (otherwise, we might remove an existing attribute).
  /// \param createAttr Create the new attribute to be added.
  /// \param getExistingAttr Get an existing, matching attribute on the given
  /// declaration.
  template<typename A>
  void handleAPINotedAttribute(
         Sema &S, Decl *D, bool shouldAddAttribute,
         VersionedInfoRole role,
         llvm::function_ref<A *()> createAttr,
         llvm::function_ref<specific_attr_iterator<A>(Decl *)> getExistingAttr =
           [](Decl *decl) { return decl->specific_attr_begin<A>(); }) {
    switch (role) {
    case VersionedInfoRole::AugmentSource:
      // If we're not adding an attribute, there's nothing to do.
      if (!shouldAddAttribute) return;

      // If the attribute is already present, we're done.
      if (getExistingAttr(D) != D->specific_attr_end<A>()) return;

      // Add the attribute.
      if (auto attr = createAttr())
        D->addAttr(attr);
      break;

    case VersionedInfoRole::ReplaceSource: {
      auto end = D->specific_attr_end<A>();
      auto existing = getExistingAttr(D);
      if (existing != end) {
        // Remove the existing attribute.
        D->getAttrs().erase(existing.getCurrent());
      }

      // If we're supposed to add a new attribute, do so.
      if (shouldAddAttribute) {
        if (auto attr = createAttr()) {
          D->addAttr(attr);
        }
      }
      break;
    }

    case VersionedInfoRole::Versioned:
      // FIXME: Retain versioned attributes separately.
      break;
    }
  }
}

static void ProcessAPINotes(Sema &S, Decl *D,
                            const api_notes::CommonEntityInfo &info,
                            VersionedInfoRole role) {
  // Availability
  if (info.Unavailable) {
    handleAPINotedAttribute<UnavailableAttr>(S, D, true, role,
      [&] {
        return UnavailableAttr::CreateImplicit(S.Context,
                                               CopyString(S.Context,
                                                          info.UnavailableMsg));
    });
  }

  if (info.UnavailableInSwift) {
    handleAPINotedAttribute<AvailabilityAttr>(S, D, true, role, [&] {
      return AvailabilityAttr::CreateImplicit(
                   S.Context,
                   &S.Context.Idents.get("swift"),
                   VersionTuple(),
                   VersionTuple(),
                   VersionTuple(),
                   /*Unavailable=*/true,
                   CopyString(S.Context, info.UnavailableMsg),
                   /*Strict=*/false,
                   /*Replacement=*/StringRef());
    },
    [](Decl *decl) {
      auto existing = decl->specific_attr_begin<AvailabilityAttr>(),
        end = decl->specific_attr_end<AvailabilityAttr>();
      while (existing != end) {
        if (auto platform = (*existing)->getPlatform()) {
          if (platform->isStr("swift"))
            break;
        }

        ++existing;
      }

      return existing;
    });
  }

  // swift_private
  if (auto swiftPrivate = info.isSwiftPrivate()) {
    handleAPINotedAttribute<SwiftPrivateAttr>(S, D, *swiftPrivate, role, [&] {
      return SwiftPrivateAttr::CreateImplicit(S.Context);
    });
  }

  // swift_name
  if (!info.SwiftName.empty()) {
    handleAPINotedAttribute<SwiftNameAttr>(S, D, true, role,
                                           [&]() -> SwiftNameAttr * {
      auto &APINoteName = S.getASTContext().Idents.get("SwiftName API Note");
      
      if (!S.DiagnoseSwiftName(D, info.SwiftName, D->getLocation(),
                               &APINoteName)) {
        return nullptr;
      }

      return SwiftNameAttr::CreateImplicit(S.Context,
                                           CopyString(S.Context,
                                                      info.SwiftName));
    });
  }
}

static void ProcessAPINotes(Sema &S, Decl *D,
                            const api_notes::CommonTypeInfo &info,
                            VersionedInfoRole role) {
  // swift_bridge
  if (auto swiftBridge = info.getSwiftBridge()) {
    handleAPINotedAttribute<SwiftBridgeAttr>(S, D, !swiftBridge->empty(), role,
                                             [&] {
      return SwiftBridgeAttr::CreateImplicit(S.Context,
                                             CopyString(S.Context,
                                                        *swiftBridge));
    });
  }

  // ns_error_domain
  if (auto nsErrorDomain = info.getNSErrorDomain()) {
    handleAPINotedAttribute<NSErrorDomainAttr>(S, D, !nsErrorDomain->empty(),
                                               role, [&] {
      return NSErrorDomainAttr::CreateImplicit(
               S.Context,
               &S.Context.Idents.get(*nsErrorDomain));
    });
  }

  ProcessAPINotes(S, D, static_cast<const api_notes::CommonEntityInfo &>(info),
                  role);
}

/// Process API notes for a variable or property.
static void ProcessAPINotes(Sema &S, Decl *D,
                            const api_notes::VariableInfo &info,
                            VersionedInfoRole role) {
  // Nullability.
  if (auto Nullability = info.getNullability()) {
    applyNullability(S, D, *Nullability, role);
  }

  // Handle common entity information.
  ProcessAPINotes(S, D, static_cast<const api_notes::CommonEntityInfo &>(info),
                  role);
}

/// Process API notes for a parameter.
static void ProcessAPINotes(Sema &S, ParmVarDecl *D,
                            const api_notes::ParamInfo &info,
                            VersionedInfoRole role) {
  // noescape
  if (auto noescape = info.isNoEscape()) {
    handleAPINotedAttribute<NoEscapeAttr>(S, D, *noescape, role, [&] {
      return NoEscapeAttr::CreateImplicit(S.Context);
    });
  }

  // Handle common entity information.
  ProcessAPINotes(S, D, static_cast<const api_notes::VariableInfo &>(info),
                  role);
}

/// Process API notes for a global variable.
static void ProcessAPINotes(Sema &S, VarDecl *D,
                            const api_notes::GlobalVariableInfo &info,
                            VersionedInfoRole role) {
  // Handle common entity information.
  ProcessAPINotes(S, D, static_cast<const api_notes::VariableInfo &>(info),
                  role);
}

/// Process API notes for an Objective-C property.
static void ProcessAPINotes(Sema &S, ObjCPropertyDecl *D,
                            const api_notes::ObjCPropertyInfo &info,
                            VersionedInfoRole role) {
  // Handle common entity information.
  ProcessAPINotes(S, D, static_cast<const api_notes::VariableInfo &>(info),
                  role);
  if (auto asAccessors = info.getSwiftImportAsAccessors()) {
    handleAPINotedAttribute<SwiftImportPropertyAsAccessorsAttr>(S, D,
                                                                *asAccessors,
                                                                role, [&] {
      return SwiftImportPropertyAsAccessorsAttr::CreateImplicit(S.Context);
    });
  }
}

namespace {
  typedef llvm::PointerUnion<FunctionDecl *, ObjCMethodDecl *> FunctionOrMethod;
}

/// Process API notes for a function or method.
static void ProcessAPINotes(Sema &S, FunctionOrMethod AnyFunc,
                            const api_notes::FunctionInfo &info,
                            VersionedInfoRole role) {
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
    applyNullability(S, D, info.getReturnTypeInfo(), role);
  }

  // Parameters.
  unsigned NumParams;
  if (FD)
    NumParams = FD->getNumParams();
  else
    NumParams = MD->param_size();
  
  for (unsigned I = 0; I != NumParams; ++I) {
    ParmVarDecl *Param;
    if (FD)
      Param = FD->getParamDecl(I);
    else
      Param = MD->param_begin()[I];
    
    // Nullability.
    if (info.NullabilityAudited)
      applyNullability(S, Param, info.getParamTypeInfo(I), role);

    if (I < info.Params.size()) {
      ProcessAPINotes(S, Param, info.Params[I], role);
    }
  }

  // Handle common entity information.
  ProcessAPINotes(S, D, static_cast<const api_notes::CommonEntityInfo &>(info),
                  role);
}

/// Process API notes for a global function.
static void ProcessAPINotes(Sema &S, FunctionDecl *D,
                            const api_notes::GlobalFunctionInfo &info,
                            VersionedInfoRole role) {

  // Handle common function information.
  ProcessAPINotes(S, FunctionOrMethod(D),
                  static_cast<const api_notes::FunctionInfo &>(info), role);
}

/// Process API notes for an enumerator.
static void ProcessAPINotes(Sema &S, EnumConstantDecl *D,
                            const api_notes::EnumConstantInfo &info,
                            VersionedInfoRole role) {

  // Handle common information.
  ProcessAPINotes(S, D,
                  static_cast<const api_notes::CommonEntityInfo &>(info),
                  role);
}

/// Process API notes for an Objective-C method.
static void ProcessAPINotes(Sema &S, ObjCMethodDecl *D,
                            const api_notes::ObjCMethodInfo &info,
                            VersionedInfoRole role) {
  // Designated initializers.
  if (info.DesignatedInit) {
    handleAPINotedAttribute<ObjCDesignatedInitializerAttr>(S, D, true, role, [&] {
      if (ObjCInterfaceDecl *IFace = D->getClassInterface()) {
        IFace->setHasDesignatedInitializers();
      }
      return ObjCDesignatedInitializerAttr::CreateImplicit(S.Context);
    });
  }

  // FIXME: This doesn't work well with versioned API notes.
  if (role == VersionedInfoRole::AugmentSource &&
      info.getFactoryAsInitKind()
        == api_notes::FactoryAsInitKind::AsClassMethod &&
      !D->getAttr<SwiftNameAttr>()) {
    D->addAttr(SwiftSuppressFactoryAsInitAttr::CreateImplicit(S.Context));
  }

  // Handle common function information.
  ProcessAPINotes(S, FunctionOrMethod(D),
                  static_cast<const api_notes::FunctionInfo &>(info), role);
}

/// Process API notes for a tag.
static void ProcessAPINotes(Sema &S, TagDecl *D,
                            const api_notes::TagInfo &info,
                            VersionedInfoRole role) {
  // Handle common type information.
  ProcessAPINotes(S, D, static_cast<const api_notes::CommonTypeInfo &>(info),
                  role);
}

/// Process API notes for a typedef.
static void ProcessAPINotes(Sema &S, TypedefNameDecl *D,
                            const api_notes::TypedefInfo &info,
                            VersionedInfoRole role) {
  // swift_wrapper
  using SwiftWrapperKind = api_notes::SwiftWrapperKind;

  if (auto swiftWrapper = info.SwiftWrapper) {
    handleAPINotedAttribute<SwiftNewtypeAttr>(S, D,
      *swiftWrapper != SwiftWrapperKind::None, role,
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
        return SwiftNewtypeAttr::CreateImplicit(
                 S.Context,
                 SwiftNewtypeAttr::GNU_swift_wrapper,
                 kind);
    });
  }

  // Handle common type information.
  ProcessAPINotes(S, D, static_cast<const api_notes::CommonTypeInfo &>(info),
                  role);
}

/// Process API notes for an Objective-C class or protocol.
static void ProcessAPINotes(Sema &S, ObjCContainerDecl *D,
                            const api_notes::ObjCContextInfo &info,
                            VersionedInfoRole role) {

  // Handle common type information.
  ProcessAPINotes(S, D, static_cast<const api_notes::CommonTypeInfo &>(info),
                  role);
}

/// Process API notes for an Objective-C class.
static void ProcessAPINotes(Sema &S, ObjCInterfaceDecl *D,
                            const api_notes::ObjCContextInfo &info,
                            VersionedInfoRole role) {
  // Handle information common to Objective-C classes and protocols.
  ProcessAPINotes(S, static_cast<clang::ObjCContainerDecl *>(D), info, role);
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
        if (auto Info = Reader->lookupGlobalVariable(VD->getName())) {
          ::ProcessAPINotes(*this, VD, *Info, Info.getSelectedRole());
        }
      }

      return;
    }

    // Global functions.
    if (auto FD = dyn_cast<FunctionDecl>(D)) {
      if (FD->getDeclName().isIdentifier()) {
        for (auto Reader : APINotes.findAPINotes(D->getLocation())) {
          if (auto Info = Reader->lookupGlobalFunction(FD->getName())) {
            ::ProcessAPINotes(*this, FD, *Info, Info.getSelectedRole());
          }
        }
      }

      return;
    }

    // Objective-C classes.
    if (auto Class = dyn_cast<ObjCInterfaceDecl>(D)) {
      for (auto Reader : APINotes.findAPINotes(D->getLocation())) {
        if (auto Info = Reader->lookupObjCClassInfo(Class->getName())) {
          ::ProcessAPINotes(*this, Class, *Info, Info.getSelectedRole());
        }
      }

      return;
    }

    // Objective-C protocols.
    if (auto Protocol = dyn_cast<ObjCProtocolDecl>(D)) {
      for (auto Reader : APINotes.findAPINotes(D->getLocation())) {
        if (auto Info = Reader->lookupObjCProtocolInfo(Protocol->getName())) {
          ::ProcessAPINotes(*this, Protocol, *Info, Info.getSelectedRole());
        }
      }

      return;
    }

    // Tags
    if (auto Tag = dyn_cast<TagDecl>(D)) {
      for (auto Reader : APINotes.findAPINotes(D->getLocation())) {
        if (auto Info = Reader->lookupTag(Tag->getName())) {
          ::ProcessAPINotes(*this, Tag, *Info, Info.getSelectedRole());
        }
      }

      return;
    }

    // Typedefs
    if (auto Typedef = dyn_cast<TypedefNameDecl>(D)) {
      for (auto Reader : APINotes.findAPINotes(D->getLocation())) {
        if (auto Info = Reader->lookupTypedef(Typedef->getName())) {
          ::ProcessAPINotes(*this, Typedef, *Info, Info.getSelectedRole());
        }
      }

      return;
    }

    return;
  }

  // Enumerators.
  if (D->getDeclContext()->getRedeclContext()->isFileContext()) {
    if (auto EnumConstant = dyn_cast<EnumConstantDecl>(D)) {
      for (auto Reader : APINotes.findAPINotes(D->getLocation())) {
        if (auto Info = Reader->lookupEnumConstant(EnumConstant->getName())) {
          ::ProcessAPINotes(*this, EnumConstant, *Info, Info.getSelectedRole());
        }
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

          if (auto Info = Reader->lookupObjCMethod(*Context, SelectorRef,
                                                   Method->isInstanceMethod())){
            ::ProcessAPINotes(*this, Method, *Info, Info.getSelectedRole());
          }
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
          if (auto Info = Reader->lookupObjCProperty(*Context,
                                                     Property->getName(),
                                                     isInstanceProperty)) {
            ::ProcessAPINotes(*this, Property, *Info, Info.getSelectedRole());
          }
        }
      }

      return;
    }

    return;
  }
}
