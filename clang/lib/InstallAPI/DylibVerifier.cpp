#include "clang/InstallAPI/DylibVerifier.h"
#include "clang/InstallAPI/FrontendRecords.h"
#include "clang/InstallAPI/InstallAPIDiagnostic.h"
#include "llvm/Demangle/Demangle.h"

using namespace llvm::MachO;

namespace clang {
namespace installapi {

/// Metadata stored about a mapping of a declaration to a symbol.
struct DylibVerifier::SymbolContext {
  // Name to use for printing in diagnostics.
  std::string PrettyPrintName{""};

  // Name to use for all querying and verification
  // purposes.
  std::string SymbolName{""};

  // Kind to map symbol type against record.
  EncodeKind Kind = EncodeKind::GlobalSymbol;

  // Frontend Attributes tied to the AST.
  const FrontendAttrs *FA = nullptr;

  // The ObjCInterface symbol type, if applicable.
  ObjCIFSymbolKind ObjCIFKind = ObjCIFSymbolKind::None;

  // Whether Decl is inlined.
  bool Inlined = false;
};

static std::string
getAnnotatedName(const Record *R, EncodeKind Kind, StringRef Name,
                 bool ValidSourceLoc = true,
                 ObjCIFSymbolKind ObjCIF = ObjCIFSymbolKind::None) {
  assert(!Name.empty() && "Need symbol name for printing");

  std::string Annotation;
  if (R->isWeakDefined())
    Annotation += "(weak-def) ";
  if (R->isWeakReferenced())
    Annotation += "(weak-ref) ";
  if (R->isThreadLocalValue())
    Annotation += "(tlv) ";

  // Check if symbol represents only part of a @interface declaration.
  const bool IsAnnotatedObjCClass = ((ObjCIF != ObjCIFSymbolKind::None) &&
                                     (ObjCIF <= ObjCIFSymbolKind::EHType));

  if (IsAnnotatedObjCClass) {
    if (ObjCIF == ObjCIFSymbolKind::EHType)
      Annotation += "Exception Type of ";
    if (ObjCIF == ObjCIFSymbolKind::MetaClass)
      Annotation += "Metaclass of ";
    if (ObjCIF == ObjCIFSymbolKind::Class)
      Annotation += "Class of ";
  }

  // Only print symbol type prefix or leading "_" if there is no source location
  // tied to it. This can only ever happen when the location has to come from
  // debug info.
  if (ValidSourceLoc) {
    if ((Kind == EncodeKind::GlobalSymbol) && Name.starts_with("_"))
      return Annotation + Name.drop_front(1).str();
    return Annotation + Name.str();
  }

  if (IsAnnotatedObjCClass)
    return Annotation + Name.str();

  switch (Kind) {
  case EncodeKind::GlobalSymbol:
    return Annotation + Name.str();
  case EncodeKind::ObjectiveCInstanceVariable:
    return Annotation + "(ObjC IVar) " + Name.str();
  case EncodeKind::ObjectiveCClass:
    return Annotation + "(ObjC Class) " + Name.str();
  case EncodeKind::ObjectiveCClassEHType:
    return Annotation + "(ObjC Class EH) " + Name.str();
  }

  llvm_unreachable("unexpected case for EncodeKind");
}

static std::string demangle(StringRef Name) {
  // InstallAPI currently only supports itanium manglings.
  if (!(Name.starts_with("_Z") || Name.starts_with("__Z") ||
        Name.starts_with("___Z")))
    return Name.str();
  char *Result = llvm::itaniumDemangle(Name);
  if (!Result)
    return Name.str();

  std::string Demangled(Result);
  free(Result);
  return Demangled;
}

static DylibVerifier::Result updateResult(const DylibVerifier::Result Prev,
                                          const DylibVerifier::Result Curr) {
  if (Prev == Curr)
    return Prev;

  // Never update from invalid or noverify state.
  if ((Prev == DylibVerifier::Result::Invalid) ||
      (Prev == DylibVerifier::Result::NoVerify))
    return Prev;

  // Don't let an ignored verification remove a valid one.
  if (Prev == DylibVerifier::Result::Valid &&
      Curr == DylibVerifier::Result::Ignore)
    return Prev;

  return Curr;
}
// __private_extern__ is a deprecated specifier that clang does not
// respect in all contexts, it should just be considered hidden for InstallAPI.
static bool shouldIgnorePrivateExternAttr(const Decl *D) {
  if (const FunctionDecl *FD = cast<FunctionDecl>(D))
    return FD->getStorageClass() == StorageClass::SC_PrivateExtern;
  if (const VarDecl *VD = cast<VarDecl>(D))
    return VD->getStorageClass() == StorageClass::SC_PrivateExtern;

  return false;
}

Record *findRecordFromSlice(const RecordsSlice *Slice, StringRef Name,
                            EncodeKind Kind) {
  switch (Kind) {
  case EncodeKind::GlobalSymbol:
    return Slice->findGlobal(Name);
  case EncodeKind::ObjectiveCInstanceVariable:
    return Slice->findObjCIVar(Name.contains('.'), Name);
  case EncodeKind::ObjectiveCClass:
  case EncodeKind::ObjectiveCClassEHType:
    return Slice->findObjCInterface(Name);
  }
  llvm_unreachable("unexpected end when finding record");
}

void DylibVerifier::updateState(Result State) {
  Ctx.FrontendState = updateResult(Ctx.FrontendState, State);
}

void DylibVerifier::addSymbol(const Record *R, SymbolContext &SymCtx,
                              TargetList &&Targets) {
  if (Targets.empty())
    Targets = {Ctx.Target};

  Exports->addGlobal(SymCtx.Kind, SymCtx.SymbolName, R->getFlags(), Targets);
}

bool DylibVerifier::shouldIgnoreObsolete(const Record *R, SymbolContext &SymCtx,
                                         const Record *DR) {
  return SymCtx.FA->Avail.isObsoleted();
}

bool DylibVerifier::compareObjCInterfaceSymbols(const Record *R,
                                                SymbolContext &SymCtx,
                                                const ObjCInterfaceRecord *DR) {
  const bool IsDeclVersionComplete =
      ((SymCtx.ObjCIFKind & ObjCIFSymbolKind::Class) ==
       ObjCIFSymbolKind::Class) &&
      ((SymCtx.ObjCIFKind & ObjCIFSymbolKind::MetaClass) ==
       ObjCIFSymbolKind::MetaClass);

  const bool IsDylibVersionComplete = DR->isCompleteInterface();

  // The common case, a complete ObjCInterface.
  if (IsDeclVersionComplete && IsDylibVersionComplete)
    return true;

  auto PrintDiagnostic = [&](auto SymLinkage, const Record *Record,
                             StringRef SymName, bool PrintAsWarning = false) {
    if (SymLinkage == RecordLinkage::Unknown)
      Ctx.emitDiag([&]() {
        Ctx.Diag->Report(SymCtx.FA->D->getLocation(),
                         PrintAsWarning ? diag::warn_library_missing_symbol
                                        : diag::err_library_missing_symbol)
            << SymName;
      });
    else
      Ctx.emitDiag([&]() {
        Ctx.Diag->Report(SymCtx.FA->D->getLocation(),
                         PrintAsWarning ? diag::warn_library_hidden_symbol
                                        : diag::err_library_hidden_symbol)
            << SymName;
      });
  };

  if (IsDeclVersionComplete) {
    // The decl represents a complete ObjCInterface, but the symbols in the
    // dylib do not. Determine which symbol is missing. To keep older projects
    // building, treat this as a warning.
    if (!DR->isExportedSymbol(ObjCIFSymbolKind::Class))
      PrintDiagnostic(DR->getLinkageForSymbol(ObjCIFSymbolKind::Class), R,
                      getAnnotatedName(R, SymCtx.Kind, SymCtx.PrettyPrintName,
                                       /*ValidSourceLoc=*/true,
                                       ObjCIFSymbolKind::Class),
                      /*PrintAsWarning=*/true);

    if (!DR->isExportedSymbol(ObjCIFSymbolKind::MetaClass))
      PrintDiagnostic(DR->getLinkageForSymbol(ObjCIFSymbolKind::MetaClass), R,
                      getAnnotatedName(R, SymCtx.Kind, SymCtx.PrettyPrintName,
                                       /*ValidSourceLoc=*/true,
                                       ObjCIFSymbolKind::MetaClass),
                      /*PrintAsWarning=*/true);
    return true;
  }

  if (DR->isExportedSymbol(SymCtx.ObjCIFKind)) {
    if (!IsDylibVersionComplete) {
      // Both the declaration and dylib have a non-complete interface.
      SymCtx.Kind = EncodeKind::GlobalSymbol;
      SymCtx.SymbolName = R->getName();
    }
    return true;
  }

  // At this point that means there was not a matching class symbol
  // to represent the one discovered as a declaration.
  PrintDiagnostic(DR->getLinkageForSymbol(SymCtx.ObjCIFKind), R,
                  SymCtx.PrettyPrintName);
  return false;
}

DylibVerifier::Result DylibVerifier::compareVisibility(const Record *R,
                                                       SymbolContext &SymCtx,
                                                       const Record *DR) {

  if (R->isExported()) {
    if (!DR) {
      Ctx.emitDiag([&]() {
        Ctx.Diag->Report(SymCtx.FA->D->getLocation(),
                         diag::err_library_missing_symbol)
            << SymCtx.PrettyPrintName;
      });
      return Result::Invalid;
    }
    if (DR->isInternal()) {
      Ctx.emitDiag([&]() {
        Ctx.Diag->Report(SymCtx.FA->D->getLocation(),
                         diag::err_library_hidden_symbol)
            << SymCtx.PrettyPrintName;
      });
      return Result::Invalid;
    }
  }

  // Emit a diagnostic for hidden declarations with external symbols, except
  // when theres an inlined attribute.
  if ((R->isInternal() && !SymCtx.Inlined) && DR && DR->isExported()) {

    if (Mode == VerificationMode::ErrorsOnly)
      return Result::Ignore;

    if (shouldIgnorePrivateExternAttr(SymCtx.FA->D))
      return Result::Ignore;

    unsigned ID;
    Result Outcome;
    if (Mode == VerificationMode::ErrorsAndWarnings) {
      ID = diag::warn_header_hidden_symbol;
      Outcome = Result::Ignore;
    } else {
      ID = diag::err_header_hidden_symbol;
      Outcome = Result::Invalid;
    }
    Ctx.emitDiag([&]() {
      Ctx.Diag->Report(SymCtx.FA->D->getLocation(), ID)
          << SymCtx.PrettyPrintName;
    });
    return Outcome;
  }

  if (R->isInternal())
    return Result::Ignore;

  return Result::Valid;
}

DylibVerifier::Result DylibVerifier::compareAvailability(const Record *R,
                                                         SymbolContext &SymCtx,
                                                         const Record *DR) {
  if (!SymCtx.FA->Avail.isUnavailable())
    return Result::Valid;

  const bool IsDeclAvailable = SymCtx.FA->Avail.isUnavailable();

  switch (Mode) {
  case VerificationMode::ErrorsAndWarnings:
    Ctx.emitDiag([&]() {
      Ctx.Diag->Report(SymCtx.FA->D->getLocation(),
                       diag::warn_header_availability_mismatch)
          << SymCtx.PrettyPrintName << IsDeclAvailable << IsDeclAvailable;
    });
    return Result::Ignore;
  case VerificationMode::Pedantic:
    Ctx.emitDiag([&]() {
      Ctx.Diag->Report(SymCtx.FA->D->getLocation(),
                       diag::err_header_availability_mismatch)
          << SymCtx.PrettyPrintName << IsDeclAvailable << IsDeclAvailable;
    });
    return Result::Invalid;
  case VerificationMode::ErrorsOnly:
    return Result::Ignore;
  case VerificationMode::Invalid:
    llvm_unreachable("Unexpected verification mode symbol verification");
  }
  llvm_unreachable("Unexpected verification mode symbol verification");
}

bool DylibVerifier::compareSymbolFlags(const Record *R, SymbolContext &SymCtx,
                                       const Record *DR) {
  std::string DisplayName =
      Demangle ? demangle(DR->getName()) : DR->getName().str();

  if (DR->isThreadLocalValue() && !R->isThreadLocalValue()) {
    Ctx.emitDiag([&]() {
      Ctx.Diag->Report(SymCtx.FA->D->getLocation(),
                       diag::err_dylib_symbol_flags_mismatch)
          << getAnnotatedName(DR, SymCtx.Kind, DisplayName)
          << DR->isThreadLocalValue();
    });
    return false;
  }
  if (!DR->isThreadLocalValue() && R->isThreadLocalValue()) {
    Ctx.emitDiag([&]() {
      SymCtx.FA->D->getLocation(),
          Ctx.Diag->Report(diag::err_header_symbol_flags_mismatch)
              << SymCtx.PrettyPrintName << R->isThreadLocalValue();
    });
    return false;
  }

  if (DR->isWeakDefined() && !R->isWeakDefined()) {
    Ctx.emitDiag([&]() {
      Ctx.Diag->Report(SymCtx.FA->D->getLocation(),
                       diag::err_dylib_symbol_flags_mismatch)
          << getAnnotatedName(DR, SymCtx.Kind, DisplayName)
          << R->isWeakDefined();
    });
    return false;
  }
  if (!DR->isWeakDefined() && R->isWeakDefined()) {
    Ctx.emitDiag([&]() {
      Ctx.Diag->Report(SymCtx.FA->D->getLocation(),
                       diag::err_header_symbol_flags_mismatch)
          << SymCtx.PrettyPrintName << R->isWeakDefined();
    });
    return false;
  }

  return true;
}

DylibVerifier::Result DylibVerifier::verifyImpl(Record *R,
                                                SymbolContext &SymCtx) {
  R->setVerify();
  if (!canVerify()) {
    // Accumulate symbols when not in verifying against dylib.
    if (R->isExported() && !SymCtx.FA->Avail.isUnavailable() &&
        !SymCtx.FA->Avail.isObsoleted()) {
      addSymbol(R, SymCtx);
    }
    return Ctx.FrontendState;
  }

  Record *DR =
      findRecordFromSlice(Ctx.DylibSlice, SymCtx.SymbolName, SymCtx.Kind);
  if (DR)
    DR->setVerify();

  if (shouldIgnoreObsolete(R, SymCtx, DR)) {
    updateState(Result::Ignore);
    return Ctx.FrontendState;
  }

  // Unavailable declarations don't need matching symbols.
  if (SymCtx.FA->Avail.isUnavailable() && (!DR || DR->isInternal())) {
    updateState(Result::Valid);
    return Ctx.FrontendState;
  }

  Result VisibilityCheck = compareVisibility(R, SymCtx, DR);
  if (VisibilityCheck != Result::Valid) {
    updateState(VisibilityCheck);
    return Ctx.FrontendState;
  }

  // All missing symbol cases to diagnose have been handled now.
  if (!DR) {
    updateState(Result::Ignore);
    return Ctx.FrontendState;
  }

  // Check for mismatching ObjC interfaces.
  if (SymCtx.ObjCIFKind != ObjCIFSymbolKind::None) {
    if (!compareObjCInterfaceSymbols(
            R, SymCtx, Ctx.DylibSlice->findObjCInterface(DR->getName()))) {
      updateState(Result::Invalid);
      return Ctx.FrontendState;
    }
  }

  Result AvailabilityCheck = compareAvailability(R, SymCtx, DR);
  if (AvailabilityCheck != Result::Valid) {
    updateState(AvailabilityCheck);
    return Ctx.FrontendState;
  }

  if (!compareSymbolFlags(R, SymCtx, DR)) {
    updateState(Result::Invalid);
    return Ctx.FrontendState;
  }

  addSymbol(R, SymCtx);
  updateState(Result::Valid);
  return Ctx.FrontendState;
}

bool DylibVerifier::canVerify() {
  return Ctx.FrontendState != Result::NoVerify;
}

void DylibVerifier::assignSlice(const Target &T) {
  assert(T == Ctx.Target && "Active targets should match.");
  if (Dylib.empty())
    return;

  // Note: there are no reexport slices with binaries, as opposed to TBD files,
  // so it can be assumed that the target match is the active top-level library.
  auto It = find_if(
      Dylib, [&T](const auto &Slice) { return T == Slice->getTarget(); });

  assert(It != Dylib.end() && "Target slice should always exist.");
  Ctx.DylibSlice = It->get();
}

void DylibVerifier::setTarget(const Target &T) {
  Ctx.Target = T;
  Ctx.DiscoveredFirstError = false;
  if (Dylib.empty()) {
    updateState(Result::NoVerify);
    return;
  }
  updateState(Result::Ignore);
  assignSlice(T);
}

DylibVerifier::Result DylibVerifier::verify(ObjCIVarRecord *R,
                                            const FrontendAttrs *FA,
                                            const StringRef SuperClass) {
  if (R->isVerified())
    return getState();

  std::string FullName =
      ObjCIVarRecord::createScopedName(SuperClass, R->getName());
  SymbolContext SymCtx{
      getAnnotatedName(R, EncodeKind::ObjectiveCInstanceVariable,
                       Demangle ? demangle(FullName) : FullName),
      FullName, EncodeKind::ObjectiveCInstanceVariable, FA};
  return verifyImpl(R, SymCtx);
}

static ObjCIFSymbolKind assignObjCIFSymbolKind(const ObjCInterfaceRecord *R) {
  ObjCIFSymbolKind Result = ObjCIFSymbolKind::None;
  if (R->getLinkageForSymbol(ObjCIFSymbolKind::Class) != RecordLinkage::Unknown)
    Result |= ObjCIFSymbolKind::Class;
  if (R->getLinkageForSymbol(ObjCIFSymbolKind::MetaClass) !=
      RecordLinkage::Unknown)
    Result |= ObjCIFSymbolKind::MetaClass;
  if (R->getLinkageForSymbol(ObjCIFSymbolKind::EHType) !=
      RecordLinkage::Unknown)
    Result |= ObjCIFSymbolKind::EHType;
  return Result;
}

DylibVerifier::Result DylibVerifier::verify(ObjCInterfaceRecord *R,
                                            const FrontendAttrs *FA) {
  if (R->isVerified())
    return getState();
  SymbolContext SymCtx;
  SymCtx.SymbolName = R->getName();
  SymCtx.ObjCIFKind = assignObjCIFSymbolKind(R);

  std::string DisplayName =
      Demangle ? demangle(SymCtx.SymbolName) : SymCtx.SymbolName;
  SymCtx.Kind = R->hasExceptionAttribute() ? EncodeKind::ObjectiveCClassEHType
                                           : EncodeKind::ObjectiveCClass;
  SymCtx.PrettyPrintName = getAnnotatedName(R, SymCtx.Kind, DisplayName);
  SymCtx.FA = FA;

  return verifyImpl(R, SymCtx);
}

DylibVerifier::Result DylibVerifier::verify(GlobalRecord *R,
                                            const FrontendAttrs *FA) {
  if (R->isVerified())
    return getState();

  // Global classifications could be obfusciated with `asm`.
  SimpleSymbol Sym = parseSymbol(R->getName());
  SymbolContext SymCtx;
  SymCtx.SymbolName = Sym.Name;
  SymCtx.PrettyPrintName =
      getAnnotatedName(R, Sym.Kind, Demangle ? demangle(Sym.Name) : Sym.Name);
  SymCtx.Kind = Sym.Kind;
  SymCtx.FA = FA;
  SymCtx.Inlined = R->isInlined();
  return verifyImpl(R, SymCtx);
}

void DylibVerifier::VerifierContext::emitDiag(
    llvm::function_ref<void()> Report) {
  if (!DiscoveredFirstError) {
    Diag->Report(diag::warn_target)
        << (PrintArch ? getArchitectureName(Target.Arch)
                      : getTargetTripleName(Target));
    DiscoveredFirstError = true;
  }

  Report();
}

} // namespace installapi
} // namespace clang
