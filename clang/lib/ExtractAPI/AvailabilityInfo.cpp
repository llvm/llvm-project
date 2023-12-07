#include "clang/ExtractAPI/AvailabilityInfo.h"
#include "clang/AST/Attr.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang;
using namespace extractapi;

AvailabilitySet::AvailabilitySet(const Decl *Decl) {
  // Collect availability attributes from all redeclrations.
  for (const auto *RD : Decl->redecls()) {
    if (const auto *A = RD->getAttr<UnavailableAttr>()) {
      if (!A->isImplicit()) {
        this->Availabilities.clear();
        UnconditionallyUnavailable = true;
      }
    }

    if (const auto *A = RD->getAttr<DeprecatedAttr>()) {
      if (!A->isImplicit()) {
        this->Availabilities.clear();
        UnconditionallyDeprecated = true;
      }
    }

    for (const auto *Attr : RD->specific_attrs<AvailabilityAttr>()) {
      StringRef Domain = Attr->getPlatform()->getName();
      auto *Availability =
          llvm::find_if(Availabilities, [Domain](const AvailabilityInfo &Info) {
            return Domain.equals(Info.Domain);
          });
      if (Availability != Availabilities.end()) {
        // Get the highest introduced version for all redeclarations.
        if (Availability->Introduced < Attr->getIntroduced())
          Availability->Introduced = Attr->getIntroduced();

        // Get the lowest deprecated version for all redeclarations.
        if (Availability->Deprecated > Attr->getDeprecated())
          Availability->Deprecated = Attr->getDeprecated();

        // Get the lowest obsoleted version for all redeclarations.
        if (Availability->Obsoleted > Attr->getObsoleted())
          Availability->Obsoleted = Attr->getObsoleted();
      } else {
        Availabilities.emplace_back(Domain, Attr->getIntroduced(),
                                    Attr->getDeprecated(), Attr->getObsoleted(),
                                    Attr->getUnavailable());
      }
    }
  }
}
