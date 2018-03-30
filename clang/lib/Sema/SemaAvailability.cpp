//===--- SemaAvailability.cpp - Availability attribute handling -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file processes the availability attribute.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaObjC.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

using namespace clang;
using namespace sema;

static bool hasMatchingEnvironmentOrNone(const ASTContext &Context,
                                         const AvailabilityAttr *AA) {
  const IdentifierInfo *IIEnvironment = AA->getEnvironment();
  auto Environment = Context.getTargetInfo().getTriple().getEnvironment();
  if (!IIEnvironment || Environment == llvm::Triple::UnknownEnvironment)
    return true;

  llvm::Triple::EnvironmentType ET =
      AvailabilityAttr::getEnvironmentType(IIEnvironment->getName());
  return Environment == ET;
}

static const AvailabilityAttr *getAttrForPlatform(ASTContext &Context,
                                                  StringRef TargetPlatform,
                                                  const Decl *D) {
  AvailabilityAttr const *PartialMatch = nullptr;
  // Check each AvailabilityAttr to find the one for this platform.
  // For multiple attributes with the same platform try to find one for this
  // environment.
  // The attribute is always on the FunctionDecl, not on the
  // FunctionTemplateDecl.
  if (const auto *FTD = dyn_cast<FunctionTemplateDecl>(D))
    D = FTD->getTemplatedDecl();
  for (const auto *A : D->attrs()) {
    if (const auto *Avail = dyn_cast<AvailabilityAttr>(A)) {
      // FIXME: this is copied from CheckAvailability. We should try to
      // de-duplicate.

      // If this attr has an inferred platform-specific attr (e.g. anyappleos
      // → ios/macos/...), use that for platform matching but return the
      // original.
      const AvailabilityAttr *EffectiveAvail = Avail->getEffectiveAttr();

      // Check if this is an App Extension "platform", and if so chop off
      // the suffix for matching with the actual platform.
      StringRef ActualPlatform = EffectiveAvail->getPlatform()->getName();
      StringRef RealizedPlatform = ActualPlatform;
      if (Context.getLangOpts().AppExt) {
        size_t suffix = RealizedPlatform.rfind("_app_extension");
        if (suffix != StringRef::npos)
          RealizedPlatform = RealizedPlatform.slice(0, suffix);
      }

      // Match the platform name.
      if (RealizedPlatform == TargetPlatform) {
        // Find the best matching attribute for this environment
        if (hasMatchingEnvironmentOrNone(Context, EffectiveAvail))
          return Avail;
        PartialMatch = Avail;
      }
    }
  }
  return PartialMatch;
}

/// The diagnostic we should emit for \c D, and the declaration that
/// originated it, or \c AR_Available.
///
/// \param D The declaration to check.
/// \param Message If non-null, this will be populated with the message from
/// the availability attribute that is selected.
/// \param ClassReceiver If we're checking the method of a class message
/// send, the class. Otherwise nullptr.
std::pair<AvailabilityResult, const NamedDecl *>
Sema::ShouldDiagnoseAvailabilityOfDecl(const NamedDecl *D, StringRef Platform,
                                       const VersionTuple &PlatformVersion,
                                       std::string *Message,
                                       ObjCInterfaceDecl *ClassReceiver) {
  AvailabilityResult Result =
      D->getAvailability(Platform, PlatformVersion, Message);

  // For typedefs, if the typedef declaration appears available look
  // to the underlying type to see if it is more restrictive.
  while (const auto *TD = dyn_cast<TypedefNameDecl>(D)) {
    if (Result != AR_Available)
      break;
    for (const Type *T = TD->getUnderlyingType().getTypePtr(); /**/; /**/) {
      if (auto *TT = dyn_cast<TagType>(T)) {
        D = TT->getDecl()->getDefinitionOrSelf();
      } else if (isa<SubstTemplateTypeParmType>(T)) {
        // A Subst* node represents a use through a template.
        // Any uses of the underlying declaration happened through it's template
        // specialization.
        goto done;
      } else {
        const Type *NextT =
            T->getLocallyUnqualifiedSingleStepDesugaredType().getTypePtr();
        if (NextT == T)
          goto done;
        T = NextT;
        continue;
      }
      /* TO_UPSTREAM(iosmac) ON*/
      Result = D->getAvailability(Platform, PlatformVersion, Message);
      /* TO_UPSTREAM(iosmac) OFF*/
      break;
    }
  }
done:
  // Forward class declarations get their attributes from their definition.
  if (const auto *IDecl = dyn_cast<ObjCInterfaceDecl>(D)) {
    if (IDecl->getDefinition()) {
      D = IDecl->getDefinition();
      Result = D->getAvailability(Platform, PlatformVersion, Message);
    }
  }

  if (const auto *ECD = dyn_cast<EnumConstantDecl>(D))
    if (Result == AR_Available) {
      const DeclContext *DC = ECD->getDeclContext();
      if (const auto *TheEnumDecl = dyn_cast<EnumDecl>(DC)) {
        Result =
            TheEnumDecl->getAvailability(Platform, PlatformVersion, Message);
        D = TheEnumDecl;
      }
    }

  // For +new, infer availability from -init.
  if (const auto *MD = dyn_cast<ObjCMethodDecl>(D)) {
    if (ObjC().NSAPIObj && ClassReceiver) {
      ObjCMethodDecl *Init = ClassReceiver->lookupInstanceMethod(
          ObjC().NSAPIObj->getInitSelector());
      if (Init && Result == AR_Available && MD->isClassMethod() &&
          MD->getSelector() == ObjC().NSAPIObj->getNewSelector() &&
          MD->definedInNSObject(getASTContext())) {
        Result = Init->getAvailability(Platform, PlatformVersion, Message);
        D = Init;
      }
    }
  }

  return {Result, D};
}

/// whether we should emit a diagnostic for \c K and \c DeclVersion in
/// the context of \c Ctx. For example, we should emit an unavailable diagnostic
/// in a deprecated context, but not the other way around.
static bool ShouldDiagnoseAvailabilityInContext(
    Sema &S, AvailabilityResult K, VersionTuple DeclVersion,
    const IdentifierInfo *DeclEnv, Decl *Ctx, const NamedDecl *OffendingDecl,
    StringRef TargetPlatform, const VersionTuple &TargetPlatformMinVersion) {
  assert(K != AR_Available && "Expected an unavailable declaration here!");

  // If this was defined using CF_OPTIONS, etc. then ignore the diagnostic.
  auto DeclLoc = Ctx->getBeginLoc();
  // This is only a problem in Foundation's C++ implementation for CF_OPTIONS.
  if (DeclLoc.isMacroID() && S.getLangOpts().CPlusPlus &&
      isa<TypedefDecl>(OffendingDecl)) {
    StringRef MacroName = S.getPreprocessor().getImmediateMacroName(DeclLoc);
    if (MacroName == "CF_OPTIONS" || MacroName == "OBJC_OPTIONS" ||
        MacroName == "SWIFT_OPTIONS" || MacroName == "NS_OPTIONS") {
      return false;
    }
  }

  // In HLSL, skip emitting diagnostic if the diagnostic mode is not set to
  // strict (-fhlsl-strict-availability), or if the target is library and the
  // availability is restricted to a specific environment/shader stage.
  // For libraries the availability will be checked later in
  // DiagnoseHLSLAvailability class once where the specific environment/shader
  // stage of the caller is known.
  // We only do this for APIs that are not explicitly deprecated. Any API that
  // is explicitly deprecated we always issue a diagnostic on.
  if (S.getLangOpts().HLSL && K != AR_Deprecated) {
    if (!S.getLangOpts().HLSLStrictAvailability ||
        (DeclEnv != nullptr &&
         S.getASTContext().getTargetInfo().getTriple().getEnvironment() ==
             llvm::Triple::EnvironmentType::Library))
      return false;
  }

  if (K == AR_Deprecated) {
    if (const auto *VD = dyn_cast<VarDecl>(OffendingDecl))
      if (VD->isLocalVarDeclOrParm() && VD->isDeprecatedInAnyTargetPlatform())
        return true;
  }

  // Checks if we should emit the availability diagnostic in the context of C.
  auto CheckContext = [&](const Decl *C) {
    if (K == AR_NotYetIntroduced) {
      if (const AvailabilityAttr *AA =
              getAttrForPlatform(S.Context, TargetPlatform, C))
        if (AA->getEffectiveIntroduced() >= DeclVersion &&
            AA->getEffectiveEnvironment() == DeclEnv)
          return true;
    } else if (K == AR_Deprecated) {
      if (C->getAvailability(TargetPlatform, TargetPlatformMinVersion) ==
          AR_Deprecated)
        return true;
    } else if (K == AR_Unavailable) {
      // It is perfectly fine to refer to an 'unavailable' Objective-C method
      // when it is referenced from within the @implementation itself. In this
      // context, we interpret unavailable as a form of access control.
      if (const auto *MD = dyn_cast<ObjCMethodDecl>(OffendingDecl)) {
        if (const auto *Impl = dyn_cast<ObjCImplDecl>(C)) {
          if (MD->getClassInterface() == Impl->getClassInterface())
            return true;
        }
      }
    }

    if (C->getAvailability(TargetPlatform, TargetPlatformMinVersion) ==
        AR_Unavailable)
      return true;
    return false;
  };

  do {
    if (CheckContext(Ctx))
      return false;

    // An implementation implicitly has the availability of the interface.
    // Unless it is "+load" method.
    if (const auto *MethodD = dyn_cast<ObjCMethodDecl>(Ctx))
      if (MethodD->isClassMethod() &&
          MethodD->getSelector().getAsString() == "load")
        return true;

    if (const auto *CatOrImpl = dyn_cast<ObjCImplDecl>(Ctx)) {
      if (const ObjCInterfaceDecl *Interface = CatOrImpl->getClassInterface())
        if (CheckContext(Interface))
          return false;
    }
    // A category implicitly has the availability of the interface.
    else if (const auto *CatD = dyn_cast<ObjCCategoryDecl>(Ctx))
      if (const ObjCInterfaceDecl *Interface = CatD->getClassInterface())
        if (CheckContext(Interface))
          return false;
  } while ((Ctx = cast_or_null<Decl>(Ctx->getDeclContext())));

  return true;
}

static unsigned getAvailabilityDiagnosticKind(
    const ASTContext &Context, const VersionTuple &DeploymentVersion,
    const VersionTuple &DeclVersion, bool HasMatchingEnv,
    bool IsTargetVariantCheck = false) {
  const auto &Triple =
      IsTargetVariantCheck
          ? *Context.getTargetInfo().getDarwinTargetVariantTriple()
          : Context.getTargetInfo().getTriple();
  VersionTuple ForceAvailabilityFromVersion;
  switch (Triple.getOS()) {
  // For iOS, emit the diagnostic even if -Wunguarded-availability is
  // not specified for deployment targets >= to iOS 11 or equivalent or
  // for declarations that were introduced in iOS 11 (macOS 10.13, ...) or
  // later.
  case llvm::Triple::IOS:
  case llvm::Triple::TvOS:
    ForceAvailabilityFromVersion = VersionTuple(/*Major=*/11);
    break;
  case llvm::Triple::WatchOS:
    ForceAvailabilityFromVersion = VersionTuple(/*Major=*/4);
    break;
  case llvm::Triple::Darwin:
  case llvm::Triple::MacOSX:
    ForceAvailabilityFromVersion = VersionTuple(/*Major=*/10, /*Minor=*/13);
    break;
  // For HLSL, use diagnostic from HLSLAvailability group which
  // are reported as errors by default and in strict diagnostic mode
  // (-fhlsl-strict-availability) and as warnings in relaxed diagnostic
  // mode (-Wno-error=hlsl-availability)
  case llvm::Triple::ShaderModel:
    return HasMatchingEnv ? diag::warn_hlsl_availability
                          : diag::warn_hlsl_availability_unavailable;
  default:
    // New Apple targets should always warn about availability.
    ForceAvailabilityFromVersion =
        (Triple.getVendor() == llvm::Triple::Apple)
            ? VersionTuple(/*Major=*/0, 0)
            : VersionTuple(/*Major=*/(unsigned)-1, (unsigned)-1);
  }
  if (DeploymentVersion >= ForceAvailabilityFromVersion ||
      DeclVersion >= ForceAvailabilityFromVersion)
    return HasMatchingEnv ? diag::warn_unguarded_availability_new
                          : diag::warn_unguarded_availability_unavailable_new;
  return HasMatchingEnv ? diag::warn_unguarded_availability
                        : diag::warn_unguarded_availability_unavailable;
}

static NamedDecl *findEnclosingDeclToAnnotate(Decl *OrigCtx) {
  for (Decl *Ctx = OrigCtx; Ctx;
       Ctx = cast_or_null<Decl>(Ctx->getDeclContext())) {
    if (isa<TagDecl>(Ctx) || isa<FunctionDecl>(Ctx) || isa<ObjCMethodDecl>(Ctx))
      return cast<NamedDecl>(Ctx);
    if (auto *CD = dyn_cast<ObjCContainerDecl>(Ctx)) {
      if (auto *Imp = dyn_cast<ObjCImplDecl>(Ctx))
        return Imp->getClassInterface();
      return CD;
    }
  }

  return dyn_cast<NamedDecl>(OrigCtx);
}

namespace {

struct AttributeInsertion {
  StringRef Prefix;
  SourceLocation Loc;
  StringRef Suffix;

  static AttributeInsertion createInsertionAfter(const NamedDecl *D) {
    return {" ", D->getEndLoc(), ""};
  }
  static AttributeInsertion createInsertionAfter(SourceLocation Loc) {
    return {" ", Loc, ""};
  }
  static AttributeInsertion createInsertionBefore(const NamedDecl *D) {
    return {"", D->getBeginLoc(), "\n"};
  }
};

} // end anonymous namespace

/// Tries to parse a string as ObjC method name.
///
/// \param Name The string to parse. Expected to originate from availability
/// attribute argument.
/// \param SlotNames The vector that will be populated with slot names. In case
/// of unsuccessful parsing can contain invalid data.
/// \returns A number of method parameters if parsing was successful,
/// std::nullopt otherwise.
static std::optional<unsigned>
tryParseObjCMethodName(StringRef Name, SmallVectorImpl<StringRef> &SlotNames,
                       const LangOptions &LangOpts) {
  // Accept replacements starting with - or + as valid ObjC method names.
  if (!Name.empty() && (Name.front() == '-' || Name.front() == '+'))
    Name = Name.drop_front(1);
  if (Name.empty())
    return std::nullopt;
  Name.split(SlotNames, ':');
  unsigned NumParams;
  if (Name.back() == ':') {
    // Remove an empty string at the end that doesn't represent any slot.
    SlotNames.pop_back();
    NumParams = SlotNames.size();
  } else {
    if (SlotNames.size() != 1)
      // Not a valid method name, just a colon-separated string.
      return std::nullopt;
    NumParams = 0;
  }
  // Verify all slot names are valid.
  bool AllowDollar = LangOpts.DollarIdents;
  for (StringRef S : SlotNames) {
    if (S.empty())
      continue;
    if (!isValidAsciiIdentifier(S, AllowDollar))
      return std::nullopt;
  }
  return NumParams;
}

/// Returns a source location in which it's appropriate to insert a new
/// attribute for the given declaration \D.
static std::optional<AttributeInsertion>
createAttributeInsertion(const NamedDecl *D, const SourceManager &SM,
                         const LangOptions &LangOpts) {
  if (isa<ObjCPropertyDecl>(D))
    return AttributeInsertion::createInsertionAfter(D);
  if (const auto *MD = dyn_cast<ObjCMethodDecl>(D)) {
    if (MD->hasBody())
      return std::nullopt;
    return AttributeInsertion::createInsertionAfter(D);
  }
  if (const auto *TD = dyn_cast<TagDecl>(D)) {
    SourceLocation Loc =
        Lexer::getLocForEndOfToken(TD->getInnerLocStart(), 0, SM, LangOpts);
    if (Loc.isInvalid())
      return std::nullopt;
    // Insert after the 'struct'/whatever keyword.
    return AttributeInsertion::createInsertionAfter(Loc);
  }
  return AttributeInsertion::createInsertionBefore(D);
}

/// Target-independent information about an availability diagnostic.
struct PlatformAgnosticAvailabilityDiagInfo {
  const NamedDecl *ReferringDecl;
  ArrayRef<SourceLocation> Locs;
  const ObjCInterfaceDecl *UnknownObjCClass;
  bool ObjCPropertyAccess;

  PlatformAgnosticAvailabilityDiagInfo(
      const NamedDecl *ReferringDecl, ArrayRef<SourceLocation> Locs,
      const ObjCInterfaceDecl *UnknownObjCClass, bool ObjCPropertyAccess)
      : ReferringDecl(ReferringDecl), Locs(Locs),
        UnknownObjCClass(UnknownObjCClass),
        ObjCPropertyAccess(ObjCPropertyAccess) {}
};

// Target-specific information about an availability diagnostic.
struct PlatformSpecificAvailabilityDiag {
  AvailabilityResult AR;
  const NamedDecl *OffendingDecl;
  std::string Message;
  const ObjCPropertyDecl *ObjCProperty;
  bool IsTargetVariantPlatform;

  PlatformSpecificAvailabilityDiag(AvailabilityResult AR,
                                   const NamedDecl *OffendingDecl,
                                   std::string Message,
                                   const ObjCPropertyDecl *ObjCProperty,
                                   bool IsTargetVariantPlatform)
      : AR(AR), OffendingDecl(OffendingDecl), Message(std::move(Message)),
        ObjCProperty(ObjCProperty),
        IsTargetVariantPlatform(IsTargetVariantPlatform) {}

  PlatformSpecificAvailabilityDiag(AvailabilityResult AR,
                                   const NamedDecl *OffendingDecl,
                                   bool IsTargetVariantPlatform)
      : AR(AR), OffendingDecl(OffendingDecl), ObjCProperty(nullptr),
        IsTargetVariantPlatform(IsTargetVariantPlatform) {}

  /// Returns the location of the note which points to the availability
  /// attribute.
  SourceLocation getNoteLocation(Sema &S) const {
    const TargetInfo &TI = S.getASTContext().getTargetInfo();
    StringRef Platform = IsTargetVariantPlatform ? TI.getTargetVariantPlatform()
                                                 : TI.getPlatformName();
    // The declaration can have multiple availability attributes, we are looking
    // at one of them.
    const AvailabilityAttr *A =
        getAttrForPlatform(S.Context, Platform, OffendingDecl);
    if (A && A->isInherited()) {
      for (const Decl *Redecl = OffendingDecl->getMostRecentDecl(); Redecl;
           Redecl = Redecl->getPreviousDecl()) {
        const AvailabilityAttr *AForRedecl =
            getAttrForPlatform(S.Context, Platform, Redecl);
        if (AForRedecl && !AForRedecl->isInherited()) {
          // If D is a declaration with inherited attributes, the note should
          // point to the declaration with actual attributes.
          return Redecl->getLocation();
        }
      }
    }
    return OffendingDecl->getLocation();
  }
};

// Check if we have to emit an availability diagnostic for a particular target.
static bool ShouldEmitAvailabilityWarning(
    Sema &S, Decl *Ctx, const PlatformAgnosticAvailabilityDiagInfo &Info,
    const PlatformSpecificAvailabilityDiag &PlatformInfo) {
  const TargetInfo &TI = S.getASTContext().getTargetInfo();
  StringRef Platform = PlatformInfo.IsTargetVariantPlatform
                           ? TI.getTargetVariantPlatform()
                           : TI.getPlatformName();
  VersionTuple PlatformMinVersion =
      PlatformInfo.IsTargetVariantPlatform
          ? TI.getTargetVariantPlatformMinVersion()
          : TI.getPlatformMinVersion();

  VersionTuple DeclVersion;
  const IdentifierInfo *IIEnv = nullptr;
  if (const AvailabilityAttr *AA =
          getAttrForPlatform(S.Context, Platform, PlatformInfo.OffendingDecl)) {
    DeclVersion = AA->getEffectiveIntroduced();
    IIEnv = AA->getEffectiveEnvironment();
  }

  return ShouldDiagnoseAvailabilityInContext(
      S, PlatformInfo.AR, DeclVersion, IIEnv, Ctx, PlatformInfo.OffendingDecl,
      Platform, PlatformMinVersion);
}

/// Emits an availability diagnostic for a not yet introduced declaration.
static void EmitNotIntroducedAvailabilityWarning(
    Sema &S, Decl *Ctx, const PlatformAgnosticAvailabilityDiagInfo &Info,
    const PlatformSpecificAvailabilityDiag &PlatformInfo,
    const PlatformSpecificAvailabilityDiag *VariantPlatformInfo = nullptr) {
  assert(PlatformInfo.AR == AR_NotYetIntroduced && "unexpected AR");
  const TargetInfo &TI = S.Context.getTargetInfo();
  StringRef TargetPlatform = PlatformInfo.IsTargetVariantPlatform
                                 ? TI.getTargetVariantPlatform()
                                 : TI.getPlatformName();
  const AvailabilityAttr *AA = getAttrForPlatform(
      S.getASTContext(), TargetPlatform, PlatformInfo.OffendingDecl);
  assert(AA != nullptr && "expecting valid availability attribute");
  VersionTuple Introduced = AA->getEffectiveIntroduced();
  bool EnvironmentMatchesOrNone =
      hasMatchingEnvironmentOrNone(S.getASTContext(), AA->getEffectiveAttr());

  std::string PlatformName(
      AvailabilityAttr::getPrettyPlatformName(TargetPlatform));
  llvm::StringRef TargetEnvironment(
      llvm::Triple::getEnvironmentTypeName(TI.getTriple().getEnvironment()));
  llvm::StringRef AttrEnvironment =
      AA->getEnvironment() ? AA->getEnvironment()->getName() : "";
  bool UseEnvironment =
      (!AttrEnvironment.empty() && !TargetEnvironment.empty());
  unsigned DiagKind;
  if (VariantPlatformInfo) {
    DiagKind = diag::warn_zippered_unguarded_availability;
  } else {
    // We would like to emit the diagnostic even if -Wunguarded-availability is
    // not specified for deployment targets >= to iOS 11 or equivalent or
    // for declarations that were introduced in iOS 11 (macOS 10.13, ...) or
    // later.
    // FIXME: Should we look at the target-variant platform here too
    // potentially?
    DiagKind = getAvailabilityDiagnosticKind(
        S.Context, S.Context.getTargetInfo().getPlatformMinVersion(),
        Introduced, EnvironmentMatchesOrNone);
  }

  VersionTuple VariantIntroduced;
  if (VariantPlatformInfo)
    VariantIntroduced =
        getAttrForPlatform(S.getASTContext(), TI.getTargetVariantPlatform(),
                           VariantPlatformInfo->OffendingDecl)
            ->getIntroduced();
  {
    auto Diag = S.Diag(Info.Locs[0], DiagKind);
    Diag << Info.ReferringDecl << PlatformName << Introduced.getAsString();
    if (!VariantPlatformInfo)
      Diag << UseEnvironment << TargetEnvironment;
    else
      Diag << AvailabilityAttr::getPrettyPlatformName(
                  TI.getTargetVariantPlatform())
           << VariantIntroduced.getAsString();
  }

  S.Diag(PlatformInfo.OffendingDecl->getLocation(),
         diag::note_partial_availability_specified_here)
      << PlatformInfo.OffendingDecl << PlatformName << Introduced.getAsString()
      << (PlatformInfo.IsTargetVariantPlatform
              ? TI.getTargetVariantPlatformMinVersion()
              : TI.getPlatformMinVersion())
             .getAsString()
      << UseEnvironment << AttrEnvironment << TargetEnvironment;
  if (VariantPlatformInfo &&
      VariantPlatformInfo->OffendingDecl != PlatformInfo.OffendingDecl)
    S.Diag(VariantPlatformInfo->OffendingDecl->getLocation(),
           diag::note_partial_availability_specified_here)
        << VariantPlatformInfo->OffendingDecl
        << AvailabilityAttr::getPrettyPlatformName(
               TI.getTargetVariantPlatform())
        << VariantIntroduced.getAsString()
        << TI.getTargetVariantPlatformMinVersion().getAsString();

  // Do not offer to silence the warning or fixits for HLSL
  if (S.getLangOpts().HLSL)
    return;

  if (const auto *Enclosing = findEnclosingDeclToAnnotate(Ctx)) {
    if (const auto *TD = dyn_cast<TagDecl>(Enclosing))
      if (TD->getDeclName().isEmpty()) {
        S.Diag(TD->getLocation(),
               diag::note_decl_unguarded_availability_silence)
            << /*Anonymous*/ 1 << TD->getKindName();
        return;
      }
    auto FixitNoteDiag = S.Diag(Enclosing->getLocation(),
                                diag::note_decl_unguarded_availability_silence)
                         << /*Named*/ 0 << Enclosing;
    // Don't offer a fixit for declarations with availability attributes.
    if (Enclosing->hasAttr<AvailabilityAttr>())
      return;
    Preprocessor &PP = S.getPreprocessor();
    if (!PP.isMacroDefined("API_AVAILABLE"))
      return;
    std::optional<AttributeInsertion> Insertion = createAttributeInsertion(
        Enclosing, S.getSourceManager(), S.getLangOpts());
    if (!Insertion)
      return;

    auto GetAvailablePlatform = [&](StringRef PlatformName) -> StringRef {
      // Apple's API_AVAILABLE macro expands roughly like this.
      // API_AVAILABLE(ios(17.0))
      // __attribute__((availability(__API_AVAILABLE_PLATFORM_ios(17.0)))
      // __attribute__((availability(ios,introduced=17.0)))
      // In order to figure out which platform name to use in the API_AVAILABLE
      // macro, the associated __API_AVAILABLE_PLATFORM_ macro needs to be
      // found. The __API_AVAILABLE_PLATFORM_ macros aren't consistent about
      // using the canonical platform name, source spelling name, or one of the
      // other supported names (i.e. one of the keys in canonicalizePlatformName
      // that's neither). Check all of the supported names for a match.
      std::vector<StringRef> EquivalentPlatforms =
          AvailabilityAttr::equivalentPlatformNames(PlatformName);
      llvm::Twine MacroPrefix = "__API_AVAILABLE_PLATFORM_";
      auto AvailablePlatform =
          llvm::find_if(EquivalentPlatforms, [&](StringRef EquivalentPlatform) {
            return PP.isMacroDefined((MacroPrefix + EquivalentPlatform).str());
          });
      if (AvailablePlatform == EquivalentPlatforms.end())
        return {};
      return *AvailablePlatform;
    };

    StringRef PlatformName = GetAvailablePlatform(TargetPlatform);
    if (PlatformName.empty())
      return;

    std::string FixItText;
    llvm::raw_string_ostream OS(FixItText);
    OS << Insertion->Prefix << "API_AVAILABLE(" << PlatformName << '('
       << Introduced.getAsString() << ')';
    if (VariantPlatformInfo) {
      StringRef VariantPlatformName =
          GetAvailablePlatform(TI.getTargetVariantPlatform());
      if (!VariantPlatformName.empty())
        OS << ", " << VariantPlatformName << '('
           << VariantIntroduced.getAsString() << ')';
    }
    OS << ')' << Insertion->Suffix;
    FixitNoteDiag << FixItHint::CreateInsertion(Insertion->Loc, OS.str());
  }
}

static void EmitZipperedNotIntroducedAvailabilityWarning(
    Sema &S, Decl *Ctx, const PlatformAgnosticAvailabilityDiagInfo &Info,
    const PlatformSpecificAvailabilityDiag &PlatformInfo,
    const PlatformSpecificAvailabilityDiag &VariantPlatformInfo) {
  EmitNotIntroducedAvailabilityWarning(S, Ctx, Info, PlatformInfo,
                                       &VariantPlatformInfo);
}

// Information about the diagnostic for a 'deprecated' or 'unavailable'.
struct UnavailableDeprecatedAvailabilityDiag {
  // Diagnostics for deprecated or unavailable.
  unsigned diag, diag_message, diag_fwdclass_message;
  unsigned diag_available_here = diag::note_availability_specified_here;

  // Matches 'diag::note_property_attribute' options.
  unsigned property_note_select;

  // Matches diag::note_availability_specified_here.
  unsigned available_here_select_kind;

  // An optional note location for the note.
  SourceLocation NoteLocation;

  CharSourceRange UseRange;
  StringRef Replacement;

  UnavailableDeprecatedAvailabilityDiag(
      Sema &S, const PlatformAgnosticAvailabilityDiagInfo &Info,
      const PlatformSpecificAvailabilityDiag &PlatformInfo) {
    const TargetInfo &TI = S.getASTContext().getTargetInfo();
    StringRef TargetPlatform = PlatformInfo.IsTargetVariantPlatform
                                   ? TI.getTargetVariantPlatform()
                                   : TI.getPlatformName();
    Replacement = "";
    UseRange = CharSourceRange();
    if (PlatformInfo.AR == AR_Deprecated) {
      if (auto AL = PlatformInfo.OffendingDecl->getAttr<DeprecatedAttr>())
        Replacement = AL->getReplacement();
      if (auto AL = getAttrForPlatform(S.Context, TargetPlatform,
                                       PlatformInfo.OffendingDecl))
        Replacement = AL->getReplacement();

      if (!Replacement.empty())
        UseRange = CharSourceRange::getCharRange(
            Info.Locs[0], S.getLocForEndOfToken(Info.Locs[0]));
    }

    if (PlatformInfo.AR == AR_Deprecated) {
      if (Info.ObjCPropertyAccess)
        diag = diag::warn_property_method_deprecated;
      else if (S.currentEvaluationContext().IsCaseExpr)
        diag = diag::warn_deprecated_switch_case;
      else
        diag = diag::warn_deprecated;

      diag_message = diag::warn_deprecated_message;
      diag_fwdclass_message = diag::warn_deprecated_fwdclass_message;
      property_note_select = /* deprecated */ 0;
      available_here_select_kind = /* deprecated */ 2;
      if (const auto *AL =
              PlatformInfo.OffendingDecl->getAttr<DeprecatedAttr>())
        NoteLocation = AL->getLocation();
      else
        NoteLocation = PlatformInfo.getNoteLocation(S);
      return;
    }
    assert(PlatformInfo.AR == AR_Unavailable && "unexpected AR");

    diag = !Info.ObjCPropertyAccess ? diag::err_unavailable
                                    : diag::err_property_method_unavailable;
    diag_message = diag::err_unavailable_message;
    diag_fwdclass_message = diag::warn_unavailable_fwdclass_message;
    property_note_select = /* unavailable */ 1;
    available_here_select_kind = /* unavailable */ 0;
    NoteLocation = PlatformInfo.getNoteLocation(S);

    if (auto AL = PlatformInfo.OffendingDecl->getAttr<UnavailableAttr>()) {
      if (AL->isImplicit() && AL->getImplicitReason()) {
        // Most of these failures are due to extra restrictions in ARC;
        // reflect that in the primary diagnostic when applicable.
        auto flagARCError = [&] {
          if (S.getLangOpts().ObjCAutoRefCount &&
              S.getSourceManager().isInSystemHeader(
                  PlatformInfo.OffendingDecl->getLocation()))
            diag = diag::err_unavailable_in_arc;
        };

        switch (AL->getImplicitReason()) {
        case UnavailableAttr::IR_None:
          break;

        case UnavailableAttr::IR_ARCForbiddenType:
          flagARCError();
          diag_available_here = diag::note_arc_forbidden_type;
          break;

        case UnavailableAttr::IR_ForbiddenWeak:
          if (S.getLangOpts().ObjCWeakRuntime)
            diag_available_here = diag::note_arc_weak_disabled;
          else
            diag_available_here = diag::note_arc_weak_no_runtime;
          break;

        case UnavailableAttr::IR_ARCForbiddenConversion:
          flagARCError();
          diag_available_here = diag::note_performs_forbidden_arc_conversion;
          break;

        case UnavailableAttr::IR_ARCInitReturnsUnrelated:
          flagARCError();
          diag_available_here = diag::note_arc_init_returns_unrelated;
          break;

        case UnavailableAttr::IR_ARCFieldWithOwnership:
          flagARCError();
          diag_available_here = diag::note_arc_field_with_ownership;
          break;
        }
      }
    }
  }

  bool operator==(const UnavailableDeprecatedAvailabilityDiag &Other) {
    return diag == Other.diag && diag_message == Other.diag_message &&
           diag_fwdclass_message == Other.diag_fwdclass_message &&
           diag_available_here == Other.diag_available_here;
  }
};

static void EmitUnavailableDeprecatedAvailabilityWarning(
    Sema &S, StringRef Message,
    const UnavailableDeprecatedAvailabilityDiag &Diag,
    const PlatformAgnosticAvailabilityDiagInfo &Info,
    const PlatformSpecificAvailabilityDiag &PlatformInfo,
    const UnavailableDeprecatedAvailabilityDiag *VariantDiag = nullptr,
    const PlatformSpecificAvailabilityDiag *VariantPlatformInfo = nullptr) {
  // Create the fix-it only when the replacement on both platforms matches.

  SmallVector<FixItHint, 12> FixIts;
  if (Diag.UseRange.isValid() &&
      (!VariantDiag || Diag.Replacement == VariantDiag->Replacement)) {
    if (const auto *MethodDecl = dyn_cast<ObjCMethodDecl>(Info.ReferringDecl)) {
      Selector Sel = MethodDecl->getSelector();
      SmallVector<StringRef, 12> SelectorSlotNames;
      std::optional<unsigned> NumParams = tryParseObjCMethodName(
          Diag.Replacement, SelectorSlotNames, S.getLangOpts());
      if (NumParams && *NumParams == Sel.getNumArgs()) {
        assert(SelectorSlotNames.size() == Info.Locs.size());
        for (unsigned I = 0; I < Info.Locs.size(); ++I) {
          if (!Sel.getNameForSlot(I).empty()) {
            CharSourceRange NameRange = CharSourceRange::getCharRange(
                Info.Locs[I], S.getLocForEndOfToken(Info.Locs[I]));
            FixIts.push_back(
                FixItHint::CreateReplacement(NameRange, SelectorSlotNames[I]));
          } else
            FixIts.push_back(
                FixItHint::CreateInsertion(Info.Locs[I], SelectorSlotNames[I]));
        }
      } else
        FixIts.push_back(
            FixItHint::CreateReplacement(Diag.UseRange, Diag.Replacement));
    } else
      FixIts.push_back(
          FixItHint::CreateReplacement(Diag.UseRange, Diag.Replacement));
  }

  SourceLocation Loc = Info.Locs[0];

  // We emit deprecation warning for deprecated specializations
  // when their instantiation stacks originate outside
  // of a system header, even if the diagnostics is suppresed at the
  // point of definition.
  SourceLocation InstantiationLoc =
      S.getTopMostPointOfInstantiation(Info.ReferringDecl);
  bool ShouldAllowWarningInSystemHeader =
      InstantiationLoc != Loc &&
      !S.getSourceManager().isInSystemHeader(InstantiationLoc);
  struct AllowWarningInSystemHeaders {
    AllowWarningInSystemHeaders(DiagnosticsEngine &E,
                                bool AllowWarningInSystemHeaders)
        : Engine(E), Prev(E.getForceSystemWarnings()) {
      if (AllowWarningInSystemHeaders)
        Engine.setForceSystemWarnings(true);
    }
    ~AllowWarningInSystemHeaders() { Engine.setForceSystemWarnings(Prev); }

  private:
    DiagnosticsEngine &Engine;
    bool Prev;
  } SystemWarningOverrideRAII(S.getDiagnostics(),
                              ShouldAllowWarningInSystemHeader);

  auto EmitObjCPropNote = [&]() {
    if (PlatformInfo.ObjCProperty)
      S.Diag(PlatformInfo.ObjCProperty->getLocation(),
             diag::note_property_attribute)
          << PlatformInfo.ObjCProperty->getDeclName()
          << Diag.property_note_select;
    if (VariantPlatformInfo && VariantPlatformInfo->ObjCProperty)
      S.Diag(VariantPlatformInfo->ObjCProperty->getLocation(),
             diag::note_property_attribute)
          << VariantPlatformInfo->ObjCProperty->getDeclName()
          << VariantDiag->property_note_select;
  };
  if (!Message.empty()) {
    S.Diag(Loc, Diag.diag_message) << Info.ReferringDecl << Message << FixIts;
    EmitObjCPropNote();
  } else if (!Info.UnknownObjCClass) {
    S.Diag(Loc, Diag.diag) << Info.ReferringDecl << FixIts;
    EmitObjCPropNote();
  } else {
    S.Diag(Loc, Diag.diag_fwdclass_message) << Info.ReferringDecl << FixIts;
    S.Diag(Info.UnknownObjCClass->getLocation(), diag::note_forward_class);
  }

  S.Diag(Diag.NoteLocation, Diag.diag_available_here)
      << PlatformInfo.OffendingDecl << Diag.available_here_select_kind;
  if (VariantDiag && VariantDiag->NoteLocation != Diag.NoteLocation)
    S.Diag(VariantDiag->NoteLocation, VariantDiag->diag_available_here)
        << VariantPlatformInfo->OffendingDecl
        << VariantDiag->available_here_select_kind;
}

/// Actually emit an availability diagnostic for a reference to an unavailable
/// decl.
///
/// \param Ctx The context that the reference occurred in
/// \param ReferringDecl The exact declaration that was referenced.
/// \param OffendingDecl A related decl to \c ReferringDecl that has an
/// availability attribute corresponding to \c K attached to it. Note that this
/// may not be the same as ReferringDecl, i.e. if an EnumDecl is annotated and
/// we refer to a member EnumConstantDecl, ReferringDecl is the EnumConstantDecl
/// and OffendingDecl is the EnumDecl.
static void DoEmitAvailabilityWarning(
    Sema &S, Decl *Ctx, const PlatformAgnosticAvailabilityDiagInfo &Info,
    const PlatformSpecificAvailabilityDiag &PlatformInfo) {
  if (!ShouldEmitAvailabilityWarning(S, Ctx, Info, PlatformInfo))
    return;

  if (PlatformInfo.AR == AR_NotYetIntroduced)
    return EmitNotIntroducedAvailabilityWarning(S, Ctx, Info, PlatformInfo);
  assert(PlatformInfo.AR != AR_Available &&
         "expected an unavailable/deprecated AR");

  if (PlatformInfo.AR == AR_Deprecated)
    // Suppress -Wdeprecated-declarations in implicit
    // functions.
    if (const auto *FD = dyn_cast_or_null<FunctionDecl>(S.getCurFunctionDecl());
        FD && FD->isImplicit())
      return;

  UnavailableDeprecatedAvailabilityDiag Diag(S, Info, PlatformInfo);
  EmitUnavailableDeprecatedAvailabilityWarning(S, PlatformInfo.Message, Diag,
                                               Info, PlatformInfo);
}

static void DoEmitZipperedAvailabilityWarning(
    Sema &S, Decl *Ctx, const PlatformAgnosticAvailabilityDiagInfo &Info,
    const PlatformSpecificAvailabilityDiag &PlatformInfo,
    const PlatformSpecificAvailabilityDiag &VariantPlatformInfo) {
  assert(PlatformInfo.IsTargetVariantPlatform == false &&
         VariantPlatformInfo.IsTargetVariantPlatform == true &&
         "invalid zippered diag");
  bool EmitP1 = ShouldEmitAvailabilityWarning(S, Ctx, Info, PlatformInfo);
  bool EmitP2 =
      ShouldEmitAvailabilityWarning(S, Ctx, Info, VariantPlatformInfo);
  if (!EmitP1 && !EmitP2)
    return;
  if (EmitP1 && EmitP2) {
    // Check if we can merge the diagnostic into one.
    if (PlatformInfo.AR == VariantPlatformInfo.AR) {
      if (PlatformInfo.AR == AR_NotYetIntroduced) {
        EmitZipperedNotIntroducedAvailabilityWarning(S, Ctx, Info, PlatformInfo,
                                                     VariantPlatformInfo);
        return;
      }
      UnavailableDeprecatedAvailabilityDiag D1(S, Info, PlatformInfo);
      UnavailableDeprecatedAvailabilityDiag D2(S, Info, VariantPlatformInfo);
      if (D1 == D2) {
        std::string Message;
        llvm::raw_string_ostream OS(Message);
        if (!PlatformInfo.Message.empty() &&
            !VariantPlatformInfo.Message.empty())
          OS << PlatformInfo.Message << " and " << VariantPlatformInfo.Message;
        else if (!PlatformInfo.Message.empty())
          OS << PlatformInfo.Message;
        else if (!VariantPlatformInfo.Message.empty())
          OS << VariantPlatformInfo.Message;
        EmitUnavailableDeprecatedAvailabilityWarning(
            S, OS.str(), D1, Info, PlatformInfo, &D2, &VariantPlatformInfo);
        return;
      }
    }
  }
  if (EmitP1)
    DoEmitAvailabilityWarning(S, Ctx, Info, PlatformInfo);
  if (EmitP2)
    DoEmitAvailabilityWarning(S, Ctx, Info, VariantPlatformInfo);
}

void Sema::handleDelayedAvailabilityCheck(DelayedDiagnostic &DD, Decl *Ctx) {
  assert(DD.Kind == DelayedDiagnostic::Availability &&
         "Expected an availability diagnostic here");

  DD.Triggered = true;
  PlatformAgnosticAvailabilityDiagInfo Info(
      DD.getAvailabilityReferringDecl(), DD.getAvailabilitySelectorLocs(),
      DD.getUnknownObjCClass(), DD.getObjCPropertyAccess());
  PlatformSpecificAvailabilityDiag PlatformInfo(
      DD.getAvailabilityResult(), DD.getAvailabilityOffendingDecl(),
      DD.getAvailabilityMessage().str(), DD.getObjCProperty(),
      DD.isTargetVariantPlatform());
  DoEmitAvailabilityWarning(*this, Ctx, Info, PlatformInfo);
}

void Sema::handleZipperedDelayedAvailabilityCheck(DelayedDiagnostic &DD,
                                                  DelayedDiagnostic &VariantDD,
                                                  Decl *Ctx) {
  assert(DD.Kind == DelayedDiagnostic::Availability &&
         VariantDD.Kind == DelayedDiagnostic::Availability &&
         "Expected an availability diagnostic here");

  DD.Triggered = true;
  VariantDD.Triggered = true;
  PlatformAgnosticAvailabilityDiagInfo Info(
      DD.getAvailabilityReferringDecl(), DD.getAvailabilitySelectorLocs(),
      DD.getUnknownObjCClass(), DD.getObjCPropertyAccess());
  PlatformSpecificAvailabilityDiag PlatformInfo(
      DD.getAvailabilityResult(), DD.getAvailabilityOffendingDecl(),
      DD.getAvailabilityMessage().str(), DD.getObjCProperty(),
      DD.isTargetVariantPlatform());
  PlatformSpecificAvailabilityDiag VariantPlatformInfo(
      VariantDD.getAvailabilityResult(),
      VariantDD.getAvailabilityOffendingDecl(),
      VariantDD.getAvailabilityMessage().str(), VariantDD.getObjCProperty(),
      VariantDD.isTargetVariantPlatform());
  DoEmitZipperedAvailabilityWarning(*this, Ctx, Info, PlatformInfo,
                                    VariantPlatformInfo);
}

namespace {

/// Returns true if the given statement can be a body-like child of \p Parent.
bool isBodyLikeChildStmt(const Stmt *S, const Stmt *Parent) {
  switch (Parent->getStmtClass()) {
  case Stmt::IfStmtClass:
    return cast<IfStmt>(Parent)->getThen() == S ||
           cast<IfStmt>(Parent)->getElse() == S;
  case Stmt::WhileStmtClass:
    return cast<WhileStmt>(Parent)->getBody() == S;
  case Stmt::DoStmtClass:
    return cast<DoStmt>(Parent)->getBody() == S;
  case Stmt::ForStmtClass:
    return cast<ForStmt>(Parent)->getBody() == S;
  case Stmt::CXXForRangeStmtClass:
    return cast<CXXForRangeStmt>(Parent)->getBody() == S;
  case Stmt::ObjCForCollectionStmtClass:
    return cast<ObjCForCollectionStmt>(Parent)->getBody() == S;
  case Stmt::CaseStmtClass:
  case Stmt::DefaultStmtClass:
    return cast<SwitchCase>(Parent)->getSubStmt() == S;
  default:
    return false;
  }
}

class StmtUSEFinder : public DynamicRecursiveASTVisitor {
  const Stmt *Target;

public:
  bool VisitStmt(Stmt *S) override { return S != Target; }

  /// Returns true if the given statement is present in the given declaration.
  static bool isContained(const Stmt *Target, const Decl *D) {
    StmtUSEFinder Visitor;
    Visitor.Target = Target;
    return !Visitor.TraverseDecl(const_cast<Decl *>(D));
  }
};

/// Traverses the AST and finds the last statement that used a given
/// declaration.
class LastDeclUSEFinder : public DynamicRecursiveASTVisitor {
  const Decl *D;

public:
  bool VisitDeclRefExpr(DeclRefExpr *DRE) override {
    if (DRE->getDecl() == D)
      return false;
    return true;
  }

  static const Stmt *findLastStmtThatUsesDecl(const Decl *D,
                                              const CompoundStmt *Scope) {
    LastDeclUSEFinder Visitor;
    Visitor.D = D;
    for (const Stmt *S : llvm::reverse(Scope->body())) {
      if (!Visitor.TraverseStmt(const_cast<Stmt *>(S)))
        return S;
    }
    return nullptr;
  }
};

/// This class implements -Wunguarded-availability.
///
/// This is done with a traversal of the AST of a function that makes reference
/// to a partially available declaration. Whenever we encounter an \c if of the
/// form: \c if(@available(...)), we use the version from the condition to visit
/// the then statement.
class DiagnoseUnguardedAvailability : public DynamicRecursiveASTVisitor {
  Sema &SemaRef;
  Decl *Ctx;

  /// Stack of potentially nested 'if (@available(...))'s.
  struct ZipperedVersionTuple {
    std::optional<VersionTuple> Version;
    std::optional<VersionTuple> VariantVersion;

    static ZipperedVersionTuple make(VersionTuple V) {
      return {V, std::nullopt};
    }
    static ZipperedVersionTuple makeVariant(VersionTuple V) {
      return {std::nullopt, V};
    }
    static ZipperedVersionTuple makeZippered(VersionTuple V,
                                             VersionTuple VariantV) {
      return {V, VariantV};
    }
  };
  SmallVector<ZipperedVersionTuple, 8> AvailabilityStack;
  SmallVector<const Stmt *, 16> StmtStack;

  void EmitNotYetIntroducedDiagnostic(
      NamedDecl *D, SourceRange Range,
      const PlatformSpecificAvailabilityDiag &PlatformInfo,
      const PlatformSpecificAvailabilityDiag *VariantPlatformInfo = nullptr);

  void DiagnoseDeclAvailability(NamedDecl *D, SourceRange Range,
                                ObjCInterfaceDecl *ClassReceiver = nullptr);

public:
  DiagnoseUnguardedAvailability(Sema &SemaRef, Decl *Ctx)
      : SemaRef(SemaRef), Ctx(Ctx) {
    const TargetInfo &TI = SemaRef.Context.getTargetInfo();
    AvailabilityStack.push_back(
        TI.hasTargetVariantPlatform()
            ? ZipperedVersionTuple::makeZippered(
                  TI.getPlatformMinVersion(),
                  TI.getTargetVariantPlatformMinVersion())
            : ZipperedVersionTuple::make(TI.getPlatformMinVersion()));
  }

  // Returns the OS version for the native/variant platform that's guarded by
  // the @available
  // checks.
  const VersionTuple &getGuardedVersion(bool IsTargetVariantPlatform) {
    if (!SemaRef.Context.getTargetInfo().hasTargetVariantPlatform())
      return *AvailabilityStack.back().Version;
    for (const auto &S : llvm::reverse(AvailabilityStack)) {
      if (IsTargetVariantPlatform) {
        if (S.VariantVersion)
          return *S.VariantVersion;
      } else if (S.Version)
        return *S.Version;
    }
    llvm_unreachable("missing availability version guard");
  }

  bool TraverseStmt(Stmt *S) override {
    if (!S)
      return true;
    StmtStack.push_back(S);
    bool Result = DynamicRecursiveASTVisitor::TraverseStmt(S);
    StmtStack.pop_back();
    return Result;
  }

  void IssueDiagnostics(Stmt *S) { TraverseStmt(S); }

  bool TraverseIfStmt(IfStmt *If) override;

  // for 'case X:' statements, don't bother looking at the 'X'; it can't lead
  // to any useful diagnostics.
  bool TraverseCaseStmt(CaseStmt *CS) override {
    return TraverseStmt(CS->getSubStmt());
  }

  bool VisitObjCMessageExpr(ObjCMessageExpr *Msg) override {
    if (ObjCMethodDecl *D = Msg->getMethodDecl()) {
      ObjCInterfaceDecl *ID = nullptr;
      QualType ReceiverTy = Msg->getClassReceiver();
      if (!ReceiverTy.isNull() && ReceiverTy->getAsObjCInterfaceType())
        ID = ReceiverTy->getAsObjCInterfaceType()->getInterface();

      DiagnoseDeclAvailability(
          D, SourceRange(Msg->getSelectorStartLoc(), Msg->getEndLoc()), ID);
    }
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *DRE) override {
    DiagnoseDeclAvailability(DRE->getDecl(),
                             SourceRange(DRE->getBeginLoc(), DRE->getEndLoc()));
    return true;
  }

  bool VisitMemberExpr(MemberExpr *ME) override {
    DiagnoseDeclAvailability(ME->getMemberDecl(),
                             SourceRange(ME->getBeginLoc(), ME->getEndLoc()));
    return true;
  }

  bool VisitObjCAvailabilityCheckExpr(ObjCAvailabilityCheckExpr *E) override {
    SemaRef.Diag(E->getBeginLoc(), diag::warn_at_available_unchecked_use)
        << (!SemaRef.getLangOpts().ObjC);
    return true;
  }

  bool VisitTypeLoc(TypeLoc Ty) override;
};

void DiagnoseUnguardedAvailability::EmitNotYetIntroducedDiagnostic(
    NamedDecl *D, SourceRange Range,
    const PlatformSpecificAvailabilityDiag &PlatformInfo,
    const PlatformSpecificAvailabilityDiag *VariantPlatformInfo) {
  // We would like to emit the diagnostic even if -Wunguarded-availability is
  // not specified for deployment targets >= to iOS 11 or equivalent or
  // for declarations that were introduced in iOS 11 (macOS 10.13, ...) or
  // later.
  const TargetInfo &TI = SemaRef.getASTContext().getTargetInfo();
  StringRef PlatformNameStrRef = PlatformInfo.IsTargetVariantPlatform
                                     ? TI.getTargetVariantPlatform()
                                     : TI.getPlatformName();
  auto *AA = getAttrForPlatform(SemaRef.getASTContext(), PlatformNameStrRef,
                                PlatformInfo.OffendingDecl);
  VersionTuple Introduced = AA->getEffectiveIntroduced();

  std::string PlatformName(
      AvailabilityAttr::getPrettyPlatformName(PlatformNameStrRef));
  llvm::StringRef TargetEnvironment(TI.getTriple().getEnvironmentName());
  bool EnvironmentMatchesOrNone = hasMatchingEnvironmentOrNone(
      SemaRef.getASTContext(), AA->getEffectiveAttr());
  llvm::StringRef AttrEnvironment =
      AA->getEnvironment() ? AA->getEnvironment()->getName() : "";
  bool UseEnvironment =
      (!AttrEnvironment.empty() && !TargetEnvironment.empty());

  unsigned DiagKind =
      VariantPlatformInfo
          ? diag::warn_zippered_unguarded_availability
          : getAvailabilityDiagnosticKind(
                SemaRef.Context,
                PlatformInfo.IsTargetVariantPlatform
                    ? TI.getTargetVariantPlatformMinVersion()
                    : TI.getPlatformMinVersion(),
                Introduced, EnvironmentMatchesOrNone,
                /*IsTargetVariantCheck=*/PlatformInfo.IsTargetVariantPlatform);

  VersionTuple VariantIntroduced;
  if (VariantPlatformInfo)
    VariantIntroduced = getAttrForPlatform(SemaRef.getASTContext(),
                                           TI.getTargetVariantPlatform(),
                                           VariantPlatformInfo->OffendingDecl)
                            ->getIntroduced();
  {
    auto Diag = SemaRef.Diag(Range.getBegin(), DiagKind)
                << Range << D << PlatformName << Introduced.getAsString();
    if (!VariantPlatformInfo)
      Diag << UseEnvironment << TargetEnvironment;
    else
      Diag << AvailabilityAttr::getPrettyPlatformName(
                  TI.getTargetVariantPlatform())
           << VariantIntroduced.getAsString();
  }

  SemaRef.Diag(PlatformInfo.OffendingDecl->getLocation(),
               diag::note_partial_availability_specified_here)
      << PlatformInfo.OffendingDecl << PlatformName << Introduced.getAsString()
      << (PlatformInfo.IsTargetVariantPlatform
              ? TI.getTargetVariantPlatformMinVersion()
              : TI.getPlatformMinVersion())
             .getAsString()
      << UseEnvironment << AttrEnvironment << TargetEnvironment;
  if (VariantPlatformInfo &&
      PlatformInfo.OffendingDecl != VariantPlatformInfo->OffendingDecl)
    SemaRef.Diag(VariantPlatformInfo->OffendingDecl->getLocation(),
                 diag::note_partial_availability_specified_here)
        << VariantPlatformInfo->OffendingDecl
        << AvailabilityAttr::getPrettyPlatformName(
               TI.getTargetVariantPlatform())
        << VariantIntroduced.getAsString()
        << TI.getTargetVariantPlatformMinVersion().getAsString();

  // Do not offer to silence the warning or fixits for HLSL
  if (SemaRef.getLangOpts().HLSL)
    return;

  auto FixitDiag =
      SemaRef.Diag(Range.getBegin(), diag::note_unguarded_available_silence)
      << Range << D
      << (SemaRef.getLangOpts().ObjC ? /*@available*/ 0
                                     : /*__builtin_available*/ 1);

  // Find the statement which should be enclosed in the if @available check.
  if (StmtStack.empty())
    return;
  const Stmt *StmtOfUse = StmtStack.back();
  const CompoundStmt *Scope = nullptr;
  for (const Stmt *S : llvm::reverse(StmtStack)) {
    if (const auto *CS = dyn_cast<CompoundStmt>(S)) {
      Scope = CS;
      break;
    }
    if (isBodyLikeChildStmt(StmtOfUse, S)) {
      // The declaration won't be seen outside of the statement, so we don't
      // have to wrap the uses of any declared variables in if (@available).
      // Therefore we can avoid setting Scope here.
      break;
    }
    StmtOfUse = S;
  }
  const Stmt *LastStmtOfUse = nullptr;
  if (isa<DeclStmt>(StmtOfUse) && Scope) {
    for (const Decl *D : cast<DeclStmt>(StmtOfUse)->decls()) {
      if (StmtUSEFinder::isContained(StmtStack.back(), D)) {
        LastStmtOfUse = LastDeclUSEFinder::findLastStmtThatUsesDecl(D, Scope);
        break;
      }
    }
  }

  const SourceManager &SM = SemaRef.getSourceManager();
  SourceLocation IfInsertionLoc = SM.getExpansionLoc(StmtOfUse->getBeginLoc());
  SourceLocation StmtEndLoc =
      SM.getExpansionRange(
            (LastStmtOfUse ? LastStmtOfUse : StmtOfUse)->getEndLoc())
          .getEnd();
  if (SM.getFileID(IfInsertionLoc) != SM.getFileID(StmtEndLoc))
    return;

  StringRef Indentation = Lexer::getIndentationForLine(IfInsertionLoc, SM);
  const char *ExtraIndentation = "    ";
  std::string FixItString;
  llvm::raw_string_ostream FixItOS(FixItString);
  StringRef FixItPlatformName;
  VersionTuple FixItVersion;

  if (AA->getInferredAttr()) {
    FixItPlatformName = "anyAppleOS";
    FixItVersion = AA->getIntroduced();
  } else {
    FixItPlatformName =
        AvailabilityAttr::getPlatformNameSourceSpelling(PlatformName);
    FixItVersion = Introduced;
  }
  FixItOS << "if ("
          << (SemaRef.getLangOpts().ObjC ? "@available" : "__builtin_available")
          << "(" << FixItPlatformName << " " << FixItVersion.getAsString();
  if (VariantPlatformInfo)
    FixItOS << ", "
            << AvailabilityAttr::getPlatformNameSourceSpelling(
                   TI.getTargetVariantPlatform())
            << " " << VariantIntroduced.getAsString();
  FixItOS << ", *)) {\n" << Indentation << ExtraIndentation;
  FixitDiag << FixItHint::CreateInsertion(IfInsertionLoc, FixItOS.str());
  SourceLocation ElseInsertionLoc = Lexer::findLocationAfterToken(
      StmtEndLoc, tok::semi, SM, SemaRef.getLangOpts(),
      /*SkipTrailingWhitespaceAndNewLine=*/false);
  if (ElseInsertionLoc.isInvalid())
    ElseInsertionLoc =
        Lexer::getLocForEndOfToken(StmtEndLoc, 0, SM, SemaRef.getLangOpts());
  FixItOS.str().clear();
  FixItOS << "\n"
          << Indentation << "} else {\n"
          << Indentation << ExtraIndentation
          << "// Fallback on earlier versions\n"
          << Indentation << "}";
  FixitDiag << FixItHint::CreateInsertion(ElseInsertionLoc, FixItOS.str());
}

void DiagnoseUnguardedAvailability::DiagnoseDeclAvailability(
    NamedDecl *D, SourceRange Range, ObjCInterfaceDecl *ReceiverClass) {
  const TargetInfo &TI = SemaRef.getASTContext().getTargetInfo();

  auto CreatePlatformDiagInfo = [&](bool IsTargetVariantPlatform)
      -> std::optional<PlatformSpecificAvailabilityDiag> {
    AvailabilityResult Result;
    const NamedDecl *OffendingDecl;
    StringRef PlatformName = IsTargetVariantPlatform
                                 ? TI.getTargetVariantPlatform()
                                 : TI.getPlatformName();
    const VersionTuple &PlatformMinVersion =
        IsTargetVariantPlatform ? TI.getTargetVariantPlatformMinVersion()
                                : TI.getPlatformMinVersion();
    std::tie(Result, OffendingDecl) = SemaRef.ShouldDiagnoseAvailabilityOfDecl(
        D, PlatformName, PlatformMinVersion, nullptr, ReceiverClass);
    // All other diagnostic kinds have already been handled in
    // DiagnoseAvailabilityOfDecl.
    if (Result != AR_NotYetIntroduced)
      return std::nullopt;

    const AvailabilityAttr *AA = getAttrForPlatform(
        SemaRef.getASTContext(), PlatformName, OffendingDecl);
    assert(AA != nullptr && "expecting valid availability attribute");
    bool EnvironmentMatchesOrNone = hasMatchingEnvironmentOrNone(
        SemaRef.getASTContext(), AA->getEffectiveAttr());
    VersionTuple Introduced = AA->getEffectiveIntroduced();

    if (EnvironmentMatchesOrNone &&
        getGuardedVersion(IsTargetVariantPlatform) >= Introduced)
      return std::nullopt;

    // If the context of this function is less available than D, we should not
    // emit a diagnostic.
    if (!ShouldDiagnoseAvailabilityInContext(
            SemaRef, Result, Introduced, AA->getEffectiveEnvironment(), Ctx,
            OffendingDecl, PlatformName, PlatformMinVersion))
      return std::nullopt;
    return PlatformSpecificAvailabilityDiag(Result, OffendingDecl,
                                            IsTargetVariantPlatform);
  };

  std::optional<PlatformSpecificAvailabilityDiag> PlatformInfo =
      CreatePlatformDiagInfo(/*IsTargetVariantPlatform=*/false);
  std::optional<PlatformSpecificAvailabilityDiag> VariantPlatformInfo;
  if (TI.hasTargetVariantPlatform())
    VariantPlatformInfo =
        CreatePlatformDiagInfo(/*IsTargetVariantPlatform=*/true);
  if (!PlatformInfo && !VariantPlatformInfo)
    return;

  if (PlatformInfo) {
    if (VariantPlatformInfo)
      EmitNotYetIntroducedDiagnostic(D, Range, *PlatformInfo,
                                     &*VariantPlatformInfo);
    else
      EmitNotYetIntroducedDiagnostic(D, Range, *PlatformInfo);
    return;
  }
  if (VariantPlatformInfo)
    EmitNotYetIntroducedDiagnostic(D, Range, *VariantPlatformInfo);
}

bool DiagnoseUnguardedAvailability::VisitTypeLoc(TypeLoc Ty) {
  const Type *TyPtr = Ty.getTypePtr();
  SourceRange Range{Ty.getBeginLoc(), Ty.getEndLoc()};

  if (Range.isInvalid())
    return true;

  if (const auto *TT = dyn_cast<TagType>(TyPtr)) {
    TagDecl *TD = TT->getDecl()->getDefinitionOrSelf();
    DiagnoseDeclAvailability(TD, Range);

  } else if (const auto *TD = dyn_cast<TypedefType>(TyPtr)) {
    TypedefNameDecl *D = TD->getDecl();
    DiagnoseDeclAvailability(D, Range);

  } else if (const auto *ObjCO = dyn_cast<ObjCObjectType>(TyPtr)) {
    if (NamedDecl *D = ObjCO->getInterface())
      DiagnoseDeclAvailability(D, Range);
  }

  return true;
}

struct ExtractedAvailabilityExpr {
  const ObjCAvailabilityCheckExpr *E = nullptr;
  bool isNegated = false;
};

ExtractedAvailabilityExpr extractAvailabilityExpr(const Expr *IfCond) {
  const auto *E = IfCond;
  bool IsNegated = false;
  while (true) {
    E = E->IgnoreParens();
    if (const auto *AE = dyn_cast<ObjCAvailabilityCheckExpr>(E)) {
      return ExtractedAvailabilityExpr{AE, IsNegated};
    }

    const auto *UO = dyn_cast<UnaryOperator>(E);
    if (!UO || UO->getOpcode() != UO_LNot) {
      return ExtractedAvailabilityExpr{};
    }
    E = UO->getSubExpr();
    IsNegated = !IsNegated;
  }
}

bool DiagnoseUnguardedAvailability::TraverseIfStmt(IfStmt *If) {
  VersionTuple CondVersion;
  VersionTuple VariantCondVersion;
  if (auto *E = dyn_cast<ObjCAvailabilityCheckExpr>(If->getCond())) {
    CondVersion = E->getVersion();
    VariantCondVersion = E->getVariantVersion();

    bool IsStar = CondVersion.empty() && VariantCondVersion.empty();
    bool IsCondRedundant =
        CondVersion <= getGuardedVersion(/*IsTargetVariantPlatform=*/false);
    bool IsVariantCondRedundant =
        VariantCondVersion.empty() ||
        VariantCondVersion <=
            getGuardedVersion(/*IsTargetVariantPlatform=*/true);
    // If we're using the '*' case here or if this check is redundant, then we
    // use the enclosing version to check both branches.
    if (IsStar || (IsCondRedundant && IsVariantCondRedundant))
      return TraverseStmt(If->getThen()) && TraverseStmt(If->getElse());
  }

  ExtractedAvailabilityExpr IfCond = extractAvailabilityExpr(If->getCond());
  if (!IfCond.E) {
    // This isn't an availability checking 'if', we can just continue.
    return DynamicRecursiveASTVisitor::TraverseIfStmt(If);
  }
  CondVersion = IfCond.E->getVersion();
  // If we're using the '*' case here or if this check is redundant, then we
  // use the enclosing version to check both branches.
  if (CondVersion.empty() || CondVersion <= AvailabilityStack.back().Version) {
    return TraverseStmt(If->getThen()) && TraverseStmt(If->getElse());
  }

  auto *Guarded = If->getThen();
  auto *Unguarded = If->getElse();
  if (IfCond.isNegated) {
    std::swap(Guarded, Unguarded);
  }
  AvailabilityStack.push_back(ZipperedVersionTuple{
      CondVersion.empty() ? std::nullopt
                          : std::optional<VersionTuple>(CondVersion),
      VariantCondVersion.empty()
          ? std::nullopt
          : std::optional<VersionTuple>(VariantCondVersion)});
  bool ShouldContinue = TraverseStmt(Guarded);

  AvailabilityStack.pop_back();

  return ShouldContinue && TraverseStmt(Unguarded);
}
} // end anonymous namespace

void Sema::DiagnoseUnguardedAvailabilityViolations(Decl *D) {
  Stmt *Body = nullptr;

  if (auto *FD = D->getAsFunction()) {
    Body = FD->getBody();

    if (auto *CD = dyn_cast<CXXConstructorDecl>(FD))
      for (const CXXCtorInitializer *CI : CD->inits())
        DiagnoseUnguardedAvailability(*this, D).IssueDiagnostics(CI->getInit());

  } else if (auto *MD = dyn_cast<ObjCMethodDecl>(D))
    Body = MD->getBody();
  else if (auto *BD = dyn_cast<BlockDecl>(D))
    Body = BD->getBody();

  assert(Body && "Need a body here!");

  DiagnoseUnguardedAvailability(*this, D).IssueDiagnostics(Body);
}

FunctionScopeInfo *Sema::getCurFunctionAvailabilityContext() {
  if (FunctionScopes.empty())
    return nullptr;

  // Conservatively search the entire current function scope context for
  // availability violations. This ensures we always correctly analyze nested
  // classes, blocks, lambdas, etc. that may or may not be inside if(@available)
  // checks themselves.
  return FunctionScopes.front();
}

void Sema::DiagnoseAvailabilityOfDecl(NamedDecl *D,
                                      ArrayRef<SourceLocation> Locs,
                                      const ObjCInterfaceDecl *UnknownObjCClass,
                                      bool ObjCPropertyAccess,
                                      bool AvoidPartialAvailabilityChecks,
                                      ObjCInterfaceDecl *ClassReceiver) {
  auto CreateAvailabilityDiagnostic = [&](bool IsTargetVariantPlatform)
      -> std::optional<PlatformSpecificAvailabilityDiag> {
    std::string Message;
    AvailabilityResult Result;
    const NamedDecl *OffendingDecl;
    // See if this declaration is unavailable, deprecated, or partial.
    const TargetInfo &TI = getASTContext().getTargetInfo();
    StringRef Platform = IsTargetVariantPlatform ? TI.getTargetVariantPlatform()
                                                 : TI.getPlatformName();
    VersionTuple VT = IsTargetVariantPlatform
                          ? TI.getTargetVariantPlatformMinVersion()
                          : TI.getPlatformMinVersion();
    std::tie(Result, OffendingDecl) = ShouldDiagnoseAvailabilityOfDecl(
        D, Platform, VT, &Message, ClassReceiver);
    if (Result == AR_Available)
      return std::nullopt;

    if (Result == AR_NotYetIntroduced) {
      if (AvoidPartialAvailabilityChecks)
        return std::nullopt;

      // We need to know the @available context in the current function to
      // diagnose this use, let DiagnoseUnguardedAvailabilityViolations do that
      // when we're done parsing the current function.
      if (FunctionScopeInfo *Context = getCurFunctionAvailabilityContext()) {
        Context->HasPotentialAvailabilityViolations = true;
        return std::nullopt;
      }
    }

    const ObjCPropertyDecl *ObjCPDecl = nullptr;
    if (const auto *MD = dyn_cast<ObjCMethodDecl>(D)) {
      if (const ObjCPropertyDecl *PD = MD->findPropertyDecl()) {
        AvailabilityResult PDeclResult =
            PD->getAvailability(Platform, VT, nullptr);
        if (PDeclResult == Result)
          ObjCPDecl = PD;
      }
    }

    return PlatformSpecificAvailabilityDiag(Result, OffendingDecl,
                                            std::move(Message), ObjCPDecl,
                                            IsTargetVariantPlatform);
  };
  std::optional<PlatformSpecificAvailabilityDiag> TargetAvailabilityDiag =
      CreateAvailabilityDiagnostic(/*IsTargetVariantPlatform=*/false);
  std::optional<PlatformSpecificAvailabilityDiag> TargetVariantAvailabilityDiag;
  if (getASTContext().getTargetInfo().hasTargetVariantPlatform())
    TargetVariantAvailabilityDiag =
        CreateAvailabilityDiagnostic(/*IsTargetVariantPlatform=*/true);
  if (!TargetAvailabilityDiag && !TargetVariantAvailabilityDiag)
    return;

  PlatformAgnosticAvailabilityDiagInfo Info(D, Locs, UnknownObjCClass,
                                            ObjCPropertyAccess);

  // Delay if we're currently parsing a declaration.
  if (DelayedDiagnostics.shouldDelayDiagnostics()) {
    auto MakeDiag = [&](const PlatformSpecificAvailabilityDiag &PlatformInfo)
        -> DelayedDiagnostic {
      return DelayedDiagnostic::makeAvailability(
          PlatformInfo.AR, Info.Locs, Info.ReferringDecl,
          PlatformInfo.OffendingDecl, Info.UnknownObjCClass,
          PlatformInfo.ObjCProperty, PlatformInfo.Message,
          Info.ObjCPropertyAccess, PlatformInfo.IsTargetVariantPlatform);
    };
    if (TargetAvailabilityDiag)
      DelayedDiagnostics.add(MakeDiag(*TargetAvailabilityDiag));
    if (TargetVariantAvailabilityDiag)
      DelayedDiagnostics.add(MakeDiag(*TargetVariantAvailabilityDiag));
    return;
  }

  Decl *Ctx = cast<Decl>(getCurLexicalContext());
  if (TargetAvailabilityDiag) {
    if (TargetVariantAvailabilityDiag) {
      DoEmitZipperedAvailabilityWarning(*this, Ctx, Info,
                                        *TargetAvailabilityDiag,
                                        *TargetVariantAvailabilityDiag);
      return;
    }
    DoEmitAvailabilityWarning(*this, Ctx, Info, *TargetAvailabilityDiag);
  }
  if (TargetVariantAvailabilityDiag)
    DoEmitAvailabilityWarning(*this, Ctx, Info, *TargetVariantAvailabilityDiag);
}

void Sema::DiagnoseAvailabilityOfDecl(NamedDecl *D,
                                      ArrayRef<SourceLocation> Locs) {
  DiagnoseAvailabilityOfDecl(D, Locs, /*UnknownObjCClass=*/nullptr,
                             /*ObjCPropertyAccess=*/false,
                             /*AvoidPartialAvailabilityChecks=*/false,
                             /*ClassReceiver=*/nullptr);
}
