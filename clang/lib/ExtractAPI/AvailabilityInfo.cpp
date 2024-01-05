#include "clang/ExtractAPI/AvailabilityInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang;
using namespace extractapi;

AvailabilityInfo::AvailabilityInfo(const Decl *Decl) {
  ASTContext &Context = Decl->getASTContext();
  StringRef PlatformName = Context.getTargetInfo().getPlatformName();

  // Collect availability attributes from all redeclarations.
  for (const auto *RD : Decl->redecls()) {
    for (const auto *A : RD->specific_attrs<AvailabilityAttr>()) {
      if (A->getPlatform()->getName() != PlatformName)
        continue;
      *this =
          AvailabilityInfo(A->getPlatform()->getName(), A->getIntroduced(),
                           A->getDeprecated(), A->getObsoleted(), false, false);
      break;
    }

    if (const auto *A = RD->getAttr<UnavailableAttr>())
      if (!A->isImplicit())
        UnconditionallyUnavailable = true;

    if (const auto *A = RD->getAttr<DeprecatedAttr>())
      if (!A->isImplicit())
        UnconditionallyDeprecated = true;
  }
}
