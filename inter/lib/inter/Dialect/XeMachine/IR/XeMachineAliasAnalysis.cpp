#include "inter/Dialect/XeMachine/IR/XeMachineAliasAnalysis.h"

#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Dialect/XeMachine/IR/XeMachineRegionFlow.h"

#include "mlir/IR/Visitors.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;
using namespace inter::xemachine;

FailureOr<RegisterAliasAnalysis>
RegisterAliasAnalysis::create(func::FuncOp function) {
  RegisterAliasAnalysis analysis;

  auto collectValue = [&](Value value) {
    if (!isa<RegType>(value.getType()))
      return;
    if (analysis.aliases.try_emplace(value).second)
      analysis.values.push_back(value);
  };
  function.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    for (Value result : operation->getResults())
      collectValue(result);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          collectValue(argument);
  });

  auto addAlias = [&](Value storage, Value alias, int64_t offsetDwords,
                      Operation *owner, bool destructive = false) {
    if (!isa<RegType>(storage.getType()) || !isa<RegType>(alias.getType()))
      return;
    analysis.aliases[storage].push_back(
        {alias, offsetDwords, owner, destructive});
    analysis.aliases[alias].push_back(
        {storage, -offsetDwords, owner, destructive});
  };

  DenseMap<int64_t, Value> architecturalRegisters;
  function.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (RegisterStorageAliasOpInterface aliasOp =
            dyn_cast<RegisterStorageAliasOpInterface>(operation)) {
      SmallVector<RegisterStorageAlias, 4> relations;
      aliasOp.getRegisterStorageAliases(relations);
      for (const RegisterStorageAlias &relation : relations)
        addAlias(relation.storage, relation.alias, relation.offset, operation,
                 relation.destructive);
    }
    if (ArchRegOp archReg = dyn_cast<ArchRegOp>(operation)) {
      Value previous = architecturalRegisters.lookup(archReg.getIndex());
      if (previous)
        addAlias(previous, archReg.getResult(), 0, operation);
      else
        architecturalRegisters.try_emplace(archReg.getIndex(),
                                           archReg.getResult());
    }
  });
  RegionFlow regionFlow(function);
  for (const RegionFlow::Branch &branch : regionFlow.getBranches())
    for (const RegionFlow::Transfer &transfer : branch.transfers)
      addAlias(transfer.operand->get(), transfer.input, 0, branch.operation);

  SmallVector<int64_t> minimumOffsets;
  SmallVector<Value, 16> pending;
  for (Value root : analysis.values) {
    if (analysis.valueInfo.count(root))
      continue;

    unsigned componentIndex = analysis.components.size();
    Component &component = analysis.components.emplace_back();
    int64_t minimumOffset = 0;
    int64_t maximumOffset = 0;
    analysis.valueInfo.try_emplace(root, ValueInfo{componentIndex, 0});
    pending.push_back(root);
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      int64_t valueOffset = analysis.valueInfo.lookup(value).offsetDwords;
      RegType type = cast<RegType>(value.getType());
      minimumOffset = std::min(minimumOffset, valueOffset);
      maximumOffset =
          std::max(maximumOffset,
                   valueOffset + static_cast<int64_t>(type.getWidthDwords()));

      if (type.getBaseGRF() >= 0) {
        int64_t originDwords =
            static_cast<int64_t>(type.getBaseGRF()) * 16 - valueOffset;
        if (component.fixedOriginDwords &&
            *component.fixedOriginDwords != originDwords)
          return function.emitError("conflicting physical register aliases");
        component.fixedOriginDwords = originDwords;
      }

      for (const Alias &alias : analysis.getAliases(value)) {
        int64_t expectedOffset = valueOffset + alias.offsetDwords;
        DenseMap<Value, ValueInfo>::iterator existing =
            analysis.valueInfo.find(alias.value);
        if (existing == analysis.valueInfo.end()) {
          analysis.valueInfo.try_emplace(
              alias.value, ValueInfo{componentIndex, expectedOffset});
          pending.push_back(alias.value);
          continue;
        }
        if (existing->second.component != componentIndex ||
            existing->second.offsetDwords != expectedOffset)
          return function.emitError("inconsistent register-storage aliases");
      }
    }

    component.widthDwords = maximumOffset - minimumOffset;
    if (component.fixedOriginDwords)
      *component.fixedOriginDwords += minimumOffset;
    minimumOffsets.push_back(minimumOffset);
  }

  for (Value value : analysis.values) {
    ValueInfo &info = analysis.valueInfo.find(value)->second;
    info.offsetDwords -= minimumOffsets[info.component];
    analysis.components[info.component].members.push_back(value);
  }

  return analysis;
}

ArrayRef<Value> RegisterAliasAnalysis::getValues() const { return values; }

ArrayRef<RegisterAliasAnalysis::Component>
RegisterAliasAnalysis::getComponents() const {
  return components;
}

const RegisterAliasAnalysis::ValueInfo *
RegisterAliasAnalysis::lookup(Value value) const {
  DenseMap<Value, ValueInfo>::const_iterator found = valueInfo.find(value);
  return found == valueInfo.end() ? nullptr : &found->second;
}

ArrayRef<RegisterAliasAnalysis::Alias>
RegisterAliasAnalysis::getAliases(Value value) const {
  DenseMap<Value, SmallVector<Alias, 4>>::const_iterator found =
      aliases.find(value);
  if (found == aliases.end())
    return {};
  return found->second;
}
