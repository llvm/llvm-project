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

/// Determine whether this is a multi-level pointer type.
static bool isMultiLevelPointerType(QualType type) {
  QualType pointee = type->getPointeeType();
  if (pointee.isNull())
    return false;

  return pointee->isAnyPointerType() || pointee->isObjCObjectPointerType() ||
         pointee->isMemberPointerType();
}

// Apply nullability to the given declaration.
static void applyNullability(Sema &S, Decl *decl, NullabilityKind nullability) {
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
                                  /*implicit=*/true);
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

static void ProcessAPINotes(Sema &S, Decl *D,
                            const api_notes::CommonEntityInfo &Info) {
  // Availability
  if (Info.Unavailable && !D->hasAttr<UnavailableAttr>()) {
    D->addAttr(UnavailableAttr::CreateImplicit(S.Context,
					       CopyString(S.Context,
							  Info.UnavailableMsg)));
  }

  if (Info.UnavailableInSwift) {
    D->addAttr(AvailabilityAttr::CreateImplicit(
		             S.Context,
                 &S.Context.Idents.get("swift"),
                 VersionTuple(),
                 VersionTuple(),
                 VersionTuple(),
                 /*Unavailable=*/true,
                 CopyString(S.Context, Info.UnavailableMsg),
                 /*Strict=*/false,
                 /*Replacement=*/StringRef()));
  }

  // swift_private
  if (Info.SwiftPrivate && !D->hasAttr<SwiftPrivateAttr>()) {
    D->addAttr(SwiftPrivateAttr::CreateImplicit(S.Context));
  }

  // swift_name
  if (!Info.SwiftName.empty() && !D->hasAttr<SwiftNameAttr>()) {
    auto &APINoteName = S.getASTContext().Idents.get("SwiftName API Note");
    
    if (!S.DiagnoseSwiftName(D, Info.SwiftName, D->getLocation(),
                             &APINoteName)) {
      return;
    }
    D->addAttr(SwiftNameAttr::CreateImplicit(S.Context,
                                       CopyString(S.Context, Info.SwiftName)));
  }
}

static void ProcessAPINotes(Sema &S, Decl *D,
                            const api_notes::CommonTypeInfo &Info) {
  // swift_bridge
  if (!Info.getSwiftBridge().empty() &&
      !D->getAttr<SwiftBridgeAttr>()) {
    D->addAttr(
      SwiftBridgeAttr::CreateImplicit(S.Context,
                                      CopyString(S.Context,
                                                 Info.getSwiftBridge())));
  }

  // ns_error_domain
  if (!Info.getNSErrorDomain().empty() &&
      !D->getAttr<NSErrorDomainAttr>()) {
    D->addAttr(
      NSErrorDomainAttr::CreateImplicit(
        S.Context,
        &S.Context.Idents.get(Info.getNSErrorDomain())));
  }

  ProcessAPINotes(S, D, static_cast<const api_notes::CommonEntityInfo &>(Info));
}

/// Process API notes for a variable or property.
static void ProcessAPINotes(Sema &S, Decl *D,
                            const api_notes::VariableInfo &Info) {
  // Nullability.
  if (auto Nullability = Info.getNullability()) {
    applyNullability(S, D, *Nullability);
  }

  // Handle common entity information.
  ProcessAPINotes(S, D, static_cast<const api_notes::CommonEntityInfo &>(Info));
}

/// Process API notes for a parameter.
static void ProcessAPINotes(Sema &S, ParmVarDecl *D,
                            const api_notes::ParamInfo &Info) {
  // noescape
  if (Info.isNoEscape() && !D->getAttr<NoEscapeAttr>())
    D->addAttr(NoEscapeAttr::CreateImplicit(S.Context));

  // Handle common entity information.
  ProcessAPINotes(S, D, static_cast<const api_notes::VariableInfo &>(Info));
}

/// Process API notes for a global variable.
static void ProcessAPINotes(Sema &S, VarDecl *D,
                            const api_notes::GlobalVariableInfo &Info) {
  // Handle common entity information.
  ProcessAPINotes(S, D, static_cast<const api_notes::VariableInfo &>(Info));
}

/// Process API notes for an Objective-C property.
static void ProcessAPINotes(Sema &S, ObjCPropertyDecl *D,
                            const api_notes::ObjCPropertyInfo &Info) {
  // Handle common entity information.
  ProcessAPINotes(S, D, static_cast<const api_notes::VariableInfo &>(Info));
}

namespace {
  typedef llvm::PointerUnion<FunctionDecl *, ObjCMethodDecl *> FunctionOrMethod;
}

/// Process API notes for a function or method.
static void ProcessAPINotes(Sema &S, FunctionOrMethod AnyFunc,
                            const api_notes::FunctionInfo &Info) {
  // Find the declaration itself.
  FunctionDecl *FD = AnyFunc.dyn_cast<FunctionDecl *>();
  Decl *D = FD;
  ObjCMethodDecl *MD = 0;
  if (!D) {
    MD = AnyFunc.get<ObjCMethodDecl *>();
    D = MD;
  }

  // Nullability of return type.
  if (Info.NullabilityAudited) {
    applyNullability(S, D, Info.getReturnTypeInfo());
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
    if (Info.NullabilityAudited)
      applyNullability(S, Param, Info.getParamTypeInfo(I));

    if (I < Info.Params.size()) {
      ProcessAPINotes(S, Param, Info.Params[I]);
    }
  }

  // Handle common entity information.
  ProcessAPINotes(S, D, static_cast<const api_notes::CommonEntityInfo &>(Info));
}

/// Process API notes for a global function.
static void ProcessAPINotes(Sema &S, FunctionDecl *D,
                            const api_notes::GlobalFunctionInfo &Info) {

  // Handle common function information.
  ProcessAPINotes(S, FunctionOrMethod(D),
                  static_cast<const api_notes::FunctionInfo &>(Info));
}

/// Process API notes for an enumerator.
static void ProcessAPINotes(Sema &S, EnumConstantDecl *D,
                            const api_notes::EnumConstantInfo &Info) {

  // Handle common information.
  ProcessAPINotes(S, D,
                  static_cast<const api_notes::CommonEntityInfo &>(Info));
}

/// Process API notes for an Objective-C method.
static void ProcessAPINotes(Sema &S, ObjCMethodDecl *D,
                            const api_notes::ObjCMethodInfo &Info) {
  // Designated initializers.
  if (Info.DesignatedInit && !D->getAttr<ObjCDesignatedInitializerAttr>()) {
    if (ObjCInterfaceDecl *IFace = D->getClassInterface()) {
      D->addAttr(ObjCDesignatedInitializerAttr::CreateImplicit(S.Context));
      IFace->setHasDesignatedInitializers();
    }
  }

  if (Info.getFactoryAsInitKind()
        == api_notes::FactoryAsInitKind::AsClassMethod &&
      !D->getAttr<SwiftNameAttr>()) {
    D->addAttr(SwiftSuppressFactoryAsInitAttr::CreateImplicit(S.Context));
  }

  // Handle common function information.
  ProcessAPINotes(S, FunctionOrMethod(D),
                  static_cast<const api_notes::FunctionInfo &>(Info));
}

/// Process API notes for a tag.
static void ProcessAPINotes(Sema &S, TagDecl *D,
                            const api_notes::TagInfo &Info) {
  // Handle common type information.
  ProcessAPINotes(S, D, static_cast<const api_notes::CommonTypeInfo &>(Info));
}

/// Process API notes for a typedef.
static void ProcessAPINotes(Sema &S, TypedefNameDecl *D,
                            const api_notes::TypedefInfo &Info) {
  // Handle common type information.
  ProcessAPINotes(S, D, static_cast<const api_notes::CommonTypeInfo &>(Info));
}

/// Process API notes for an Objective-C class or protocol.
static void ProcessAPINotes(Sema &S, ObjCContainerDecl *D,
                            const api_notes::ObjCContextInfo &Info) {

  // Handle common type information.
  ProcessAPINotes(S, D, static_cast<const api_notes::CommonTypeInfo &>(Info));
}

/// Process API notes for an Objective-C class.
static void ProcessAPINotes(Sema &S, ObjCInterfaceDecl *D,
                            const api_notes::ObjCContextInfo &Info) {
  // Handle information common to Objective-C classes and protocols.
  ProcessAPINotes(S, static_cast<clang::ObjCContainerDecl *>(D), Info);
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
      if (api_notes::APINotesReader *Reader
            = APINotes.findAPINotes(D->getLocation())) {
        if (auto Info = Reader->lookupGlobalVariable(VD->getName())) {
          ::ProcessAPINotes(*this, VD, *Info);
        }
      }

      return;
    }

    // Global functions.
    if (auto FD = dyn_cast<FunctionDecl>(D)) {
      if (FD->getDeclName().isIdentifier()) {
        if (api_notes::APINotesReader *Reader
              = APINotes.findAPINotes(D->getLocation())) {
          if (auto Info = Reader->lookupGlobalFunction(FD->getName())) {
            ::ProcessAPINotes(*this, FD, *Info);
          }
        }
      }

      return;
    }

    // Objective-C classes.
    if (auto Class = dyn_cast<ObjCInterfaceDecl>(D)) {
      if (api_notes::APINotesReader *Reader
            = APINotes.findAPINotes(D->getLocation())) {
        if (auto Info = Reader->lookupObjCClass(Class->getName())) {
          ::ProcessAPINotes(*this, Class, Info->second);
        }
      }

      return;
    }

    // Objective-C protocols.
    if (auto Protocol = dyn_cast<ObjCProtocolDecl>(D)) {
      if (api_notes::APINotesReader *Reader
            = APINotes.findAPINotes(D->getLocation())) {
        if (auto Info = Reader->lookupObjCProtocol(Protocol->getName())) {
          ::ProcessAPINotes(*this, Protocol, Info->second);
        }
      }

      return;
    }

    // Tags
    if (auto Tag = dyn_cast<TagDecl>(D)) {
      if (api_notes::APINotesReader *Reader
            = APINotes.findAPINotes(D->getLocation())) {
        if (auto Info = Reader->lookupTag(Tag->getName())) {
          ::ProcessAPINotes(*this, Tag, *Info);
        }
      }

      return;
    }

    // Typedefs
    if (auto Typedef = dyn_cast<TypedefNameDecl>(D)) {
      if (api_notes::APINotesReader *Reader
            = APINotes.findAPINotes(D->getLocation())) {
        if (auto Info = Reader->lookupTypedef(Typedef->getName())) {
          ::ProcessAPINotes(*this, Typedef, *Info);
        }
      }

      return;
    }

    return;
  }

  // Enumerators.
  if (D->getDeclContext()->getRedeclContext()->isFileContext()) {
    if (auto EnumConstant = dyn_cast<EnumConstantDecl>(D)) {
      if (api_notes::APINotesReader *Reader
            = APINotes.findAPINotes(D->getLocation())) {
        if (auto Info = Reader->lookupEnumConstant(EnumConstant->getName())) {
          ::ProcessAPINotes(*this, EnumConstant, *Info);
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
        if (auto Found = Reader->lookupObjCProtocol(Protocol->getName()))
          return Found->first;

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
        if (auto Found = Reader->lookupObjCClass(Class->getName()))
          return Found->first;

        return None;

      }

      return None;
    };

    // Objective-C methods.
    if (auto Method = dyn_cast<ObjCMethodDecl>(D)) {
      if (api_notes::APINotesReader *Reader
            = APINotes.findAPINotes(D->getLocation())) {
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
            ::ProcessAPINotes(*this, Method, *Info);
          }
        }
      }
    }

    // Objective-C properties.
    if (auto Property = dyn_cast<ObjCPropertyDecl>(D)) {
      if (api_notes::APINotesReader *Reader
            = APINotes.findAPINotes(D->getLocation())) {
        if (auto Context = GetContext(Reader)) {
          if (auto Info = Reader->lookupObjCProperty(*Context,
                                                     Property->getName())) {
            ::ProcessAPINotes(*this, Property, *Info);
          }
        }
      }

      return;
    }

    return;
  }
}
