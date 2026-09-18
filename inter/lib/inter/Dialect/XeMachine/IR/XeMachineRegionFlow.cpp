#include "inter/Dialect/XeMachine/IR/XeMachineRegionFlow.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace inter::xemachine;

RegionFlow::RegionFlow(Operation *root) {
  root->walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (RegionBranchOpInterface branch =
            dyn_cast<RegionBranchOpInterface>(operation))
      build(branch);
  });
}

void RegionFlow::build(RegionBranchOpInterface branchInterface) {
  unsigned branchId = branches.size();
  Branch &branch = branches.emplace_back();
  branch.operation = branchInterface.getOperation();
  for (auto [index, region] : llvm::enumerate(branch.operation->getRegions())) {
    branch.regions.push_back(&region);
    regionLocations.try_emplace(
        &region, RegionLocation{branchId, static_cast<unsigned>(index)});
  }

  unsigned regionCount = branch.regions.size();
  branch.entryRegions.resize(regionCount);
  branch.repetitiveRegions.resize(regionCount);
  branch.reachable.resize(regionCount, llvm::BitVector(regionCount));

  for (RegionBranchPoint point : branchInterface.getAllRegionBranchPoints()) {
    SmallVector<RegionSuccessor, 4> successors;
    branchInterface.getSuccessorRegions(point, successors);
    Region *source = nullptr;
    Operation *sourceOperation = branch.operation;
    if (!point.isParent()) {
      RegionBranchTerminatorOpInterface terminator =
          point.getTerminatorPredecessorOrNull();
      source = terminator->getParentRegion();
      sourceOperation = terminator.getOperation();
    }
    for (RegionSuccessor successor : successors) {
      Region *target =
          successor.isRegion() ? successor.getSuccessor() : nullptr;
      if (!source && target)
        branch.entryRegions.set(target->getRegionNumber());
      if (source && target)
        branch.reachable[source->getRegionNumber()].set(
            target->getRegionNumber());

      OperandRange operands =
          branchInterface.getSuccessorOperands(point, successor);
      ValueRange inputs = branchInterface.getSuccessorInputs(successor);
      assert(operands.size() == inputs.size() &&
             "verified region successor arity must match");
      MutableArrayRef<OpOperand> opOperands(operands.getBase(),
                                            operands.size());
      for (auto [index, values] :
           llvm::enumerate(llvm::zip_equal(opOperands, inputs))) {
        auto [operand, input] = values;
        branch.transfers.push_back({input, &operand, source, target,
                                    sourceOperation,
                                    static_cast<unsigned>(index)});
      }
    }
  }

  for (unsigned via = 0; via < regionCount; ++via)
    for (unsigned source = 0; source < regionCount; ++source)
      if (branch.reachable[source].test(via))
        branch.reachable[source] |= branch.reachable[via];
  for (unsigned index = 0; index < regionCount; ++index)
    branch.repetitiveRegions[index] = branch.reachable[index].test(index);
  branchIds.try_emplace(branch.operation, branchId);
}

const RegionFlow::Branch *RegionFlow::lookup(Operation *operation) const {
  auto found = branchIds.find(operation);
  return found == branchIds.end() ? nullptr : &branches[found->second];
}

bool RegionFlow::isRepetitive(Region *region) const {
  auto found = regionLocations.find(region);
  if (found == regionLocations.end())
    return false;
  const RegionLocation &location = found->second;
  return branches[location.branch].repetitiveRegions.test(location.region);
}

bool RegionFlow::mayReach(Region *source, Region *target) const {
  if (!source || !target)
    return false;
  auto sourceLocation = regionLocations.find(source);
  auto targetLocation = regionLocations.find(target);
  if (sourceLocation == regionLocations.end() ||
      targetLocation == regionLocations.end() ||
      sourceLocation->second.branch != targetLocation->second.branch)
    return false;
  return branches[sourceLocation->second.branch]
      .reachable[sourceLocation->second.region]
      .test(targetLocation->second.region);
}

bool RegionFlow::areMutuallyExclusive(Region *lhs, Region *rhs) const {
  if (!lhs || !rhs || lhs == rhs)
    return false;
  auto lhsLocation = regionLocations.find(lhs);
  auto rhsLocation = regionLocations.find(rhs);
  if (lhsLocation == regionLocations.end() ||
      rhsLocation == regionLocations.end() ||
      lhsLocation->second.branch != rhsLocation->second.branch)
    return false;
  const Branch &branch = branches[lhsLocation->second.branch];
  return branch.entryRegions.test(lhsLocation->second.region) &&
         branch.entryRegions.test(rhsLocation->second.region) &&
         !mayReach(lhs, rhs) && !mayReach(rhs, lhs);
}

Region *RegionFlow::getEnclosingRepetitiveRegion(Operation *operation) const {
  for (Region *region = operation->getParentRegion(); region;
       region = region->getParentRegion())
    if (isRepetitive(region))
      return region;
  return nullptr;
}
