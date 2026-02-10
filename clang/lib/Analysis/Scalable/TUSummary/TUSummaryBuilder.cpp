#include "clang/Analysis/Scalable/TUSummary/TUSummaryBuilder.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include <memory>

using namespace clang;
using namespace ssaf;

EntityId TUSummaryBuilder::addEntity(const EntityName &E,
                                     const EntityLinkage &Linkage) {
  EntityId Id = Summary.IdTable.getId(E);
  const EntityLinkage &ExistingLinkage =
      Summary.Entities.try_emplace(Id, Linkage).first->second;
  if (ExistingLinkage != Linkage) {
    // print ExistingLinkage, Linkage, and ID;
    llvm::report_fatal_error("Entity already exists: ");
  }
  return Id;
}

void TUSummaryBuilder::addFact(EntityId ContributingEntity,
                               std::unique_ptr<EntitySummary> NewData) {
  Summary.Data[NewData->getSummaryName()][ContributingEntity] =
      std::move(NewData);
}