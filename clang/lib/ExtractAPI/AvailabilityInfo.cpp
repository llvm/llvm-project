#include "clang/ExtractAPI/AvailabilityInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang::extractapi;
using namespace llvm;

AvailabilityInfo AvailabilityInfo::createFromDecl(const Decl *Decl) {
  ASTContext &Context = Decl->getASTContext();
  StringRef PlatformName = Context.getTargetInfo().getPlatformName();
  AvailabilityInfo Availability;

  // Collect availability attributes from all redeclarations.
  for (const auto *RD : Decl->redecls()) {
    for (const auto *A : RD->specific_attrs<AvailabilityAttr>()) {
      if (A->getPlatform()->getName() != PlatformName)
        continue;
      Availability =
          AvailabilityInfo(A->getPlatform()->getName(), A->getIntroduced(),
                           A->getDeprecated(), A->getObsoleted(), false, false);
      break;
    }

    if (const auto *A = RD->getAttr<UnavailableAttr>())
      if (!A->isImplicit())
        Availability.UnconditionallyUnavailable = true;

    if (const auto *A = RD->getAttr<DeprecatedAttr>())
      if (!A->isImplicit())
        Availability.UnconditionallyDeprecated = true;
  }
  return Availability;
}
