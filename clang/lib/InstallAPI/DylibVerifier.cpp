#include "clang/InstallAPI/DylibVerifier.h"
#include "clang/InstallAPI/FrontendRecords.h"
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
  // Itanium encoding requires 1 or 3 leading underscores, followed by 'Z'.
  if (!(Name.starts_with("_Z") || Name.starts_with("___Z")))
    return Name.str();
  char *Result = llvm::itaniumDemangle(Name.data());
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

void DylibVerifier::updateState(Result State) {
  Ctx.FrontendState = updateResult(Ctx.FrontendState, State);
}

void DylibVerifier::addSymbol(const Record *R, SymbolContext &SymCtx,
                              TargetList &&Targets) {
  if (Targets.empty())
    Targets = {Ctx.Target};

  Exports->addGlobal(SymCtx.Kind, SymCtx.SymbolName, R->getFlags(), Targets);
}

DylibVerifier::Result DylibVerifier::verifyImpl(Record *R,
                                                SymbolContext &SymCtx) {
  R->setVerify();
  if (!canVerify()) {
    // Accumulate symbols when not in verifying against dylib.
    if (R->isExported() && !SymCtx.FA->Avail.isUnconditionallyUnavailable() &&
        !SymCtx.FA->Avail.isObsoleted()) {
      addSymbol(R, SymCtx);
    }
    return Ctx.FrontendState;
  }
  return Ctx.FrontendState;
}

bool DylibVerifier::canVerify() {
  return Ctx.FrontendState != Result::NoVerify;
}

void DylibVerifier::setTarget(const Target &T) {
  Ctx.Target = T;
  Ctx.DiscoveredFirstError = false;
  updateState(Dylib.empty() ? Result::NoVerify : Result::Ignore);
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
  return verifyImpl(R, SymCtx);
}

} // namespace installapi
} // namespace clang
