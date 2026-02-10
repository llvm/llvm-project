#include "clang/Analysis/Scalable/TUSummary/TUSummaryBuilder.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include <memory>

using namespace clang;
using namespace ssaf;

EntityId TUSummaryBuilder::addEntity(const EntityName &E) {
  return Summary.IdTable.getId(E);
}

void TUSummaryBuilder::addFact(EntityId ContributingEntity,
                               std::unique_ptr<EntitySummary> NewData) {
  Summary.Data[NewData->getSummaryName()][ContributingEntity] =
      std::move(NewData);
}