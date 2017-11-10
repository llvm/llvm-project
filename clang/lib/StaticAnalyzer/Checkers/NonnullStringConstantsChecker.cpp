//==- NonnullStringConstantsChecker.cpp ---------------------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This checker adds an assumption that constant string-like globals are
//  non-null, as otherwise they generally do not convey any useful information.
//  The assumption is useful, as many framework use such global const strings,
//  and the analyzer might not be able to infer the global value if the
//  definition is in a separate translation unit.
//  The following types (and their typedef aliases) are considered string-like:
//   - `char* const`
//   - `const CFStringRef` from CoreFoundation
//   - `NSString* const` from Foundation
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"

using namespace clang;
using namespace ento;

namespace {

class NonnullStringConstantsChecker : public Checker<check::Location> {
  mutable IdentifierInfo *NSStringII = nullptr;
  mutable IdentifierInfo *CFStringRefII = nullptr;

public:
  NonnullStringConstantsChecker() {}

  void checkLocation(SVal l, bool isLoad, const Stmt *S,
                     CheckerContext &C) const;

private:
  void initIdentifierInfo(ASTContext &Ctx) const;

  bool isGlobalConstString(SVal V) const;

  bool isStringlike(QualType Ty) const;
};

} // namespace

/// Lazily initialize cache for required identifier informations.
void NonnullStringConstantsChecker::initIdentifierInfo(ASTContext &Ctx) const {
  if (NSStringII)
    return;

  NSStringII = &Ctx.Idents.get("NSString");
  CFStringRefII = &Ctx.Idents.get("CFStringRef");
}

/// Add an assumption that const string-like globals are non-null.
void NonnullStringConstantsChecker::checkLocation(SVal location, bool isLoad,
                                                 const Stmt *S,
                                                 CheckerContext &C) const {
  initIdentifierInfo(C.getASTContext());
  if (!isLoad || !location.isValid())
    return;

  ProgramStateRef State = C.getState();
  SVal V = State->getSVal(location.castAs<Loc>());

  if (isGlobalConstString(location)) {
    Optional<DefinedOrUnknownSVal> Constr = V.getAs<DefinedOrUnknownSVal>();

    if (Constr) {

      // Assume that the variable is non-null.
      ProgramStateRef OutputState = State->assume(*Constr, true);
      C.addTransition(OutputState);
    }
  }
}

/// \param V loaded lvalue.
/// \return whether {@code val} is a string-like const global.
bool NonnullStringConstantsChecker::isGlobalConstString(SVal V) const {
  Optional<loc::MemRegionVal> RegionVal = V.getAs<loc::MemRegionVal>();
  if (!RegionVal)
    return false;
  auto *Region = dyn_cast<VarRegion>(RegionVal->getAsRegion());
  if (!Region)
    return false;
  const VarDecl *Decl = Region->getDecl();

  if (!Decl->hasGlobalStorage())
    return false;

  QualType Ty = Decl->getType();
  bool HasConst = Ty.isConstQualified();
  if (isStringlike(Ty) && HasConst)
    return true;

  // Look through the typedefs.
  while (auto *T = dyn_cast<TypedefType>(Ty)) {
    Ty = T->getDecl()->getUnderlyingType();

    // It is sufficient for any intermediate typedef
    // to be classified const.
    HasConst = HasConst || Ty.isConstQualified();
    if (isStringlike(Ty) && HasConst)
      return true;
  }
  return false;
}

/// \return whether {@code type} is a string-like type.
bool NonnullStringConstantsChecker::isStringlike(QualType Ty) const {

  if (Ty->isPointerType() && Ty->getPointeeType()->isCharType())
    return true;

  if (auto *T = dyn_cast<ObjCObjectPointerType>(Ty)) {
    return T->getInterfaceDecl()->getIdentifier() == NSStringII;
  } else if (auto *T = dyn_cast<TypedefType>(Ty)) {
    return T->getDecl()->getIdentifier() == CFStringRefII;
  }
  return false;
}

void ento::registerNonnullStringConstantsChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<NonnullStringConstantsChecker>();
}
