#include "clang/Analysis/Scalable/TUSummary/TUSummaryBuilder.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/TUSummary/EntitySummary.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include <memory>
#include <utility>

using namespace clang;
using namespace ssaf;

EntityId TUSummaryBuilder::addEntity(const EntityName &E) {
  return Summary.IdTable.getId(E);
}

std::pair<EntitySummary *, bool>
TUSummaryBuilder::addSummaryImpl(EntityId Entity,
                                 std::unique_ptr<EntitySummary> &&Data) {
  auto &EntitySummaries = Summary.Data[Data->getSummaryName()];
  auto [It, Inserted] = EntitySummaries.try_emplace(Entity, std::move(Data));
  return {It->second.get(), Inserted};
}
