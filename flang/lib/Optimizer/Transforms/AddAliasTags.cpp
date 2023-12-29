//===- AddAliasTags.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// Adds TBAA alias tags to fir loads and stores, based on information from
/// fir::AliasAnalysis. More are added later in CodeGen - see fir::TBAABuilder
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Analysis/TBAAForest.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FirAliasTagOpInterface.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace fir {
#define GEN_PASS_DEF_ADDALIASTAGS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "fir-add-alias-tags"

static llvm::cl::opt<bool>
    enableDummyArgs("dummy-arg-tbaa", llvm::cl::init(true), llvm::cl::Hidden,
                    llvm::cl::desc("Add TBAA tags to dummy arguments"));
static llvm::cl::opt<bool>
    enableGlobals("globals-tbaa", llvm::cl::init(true), llvm::cl::Hidden,
                  llvm::cl::desc("Add TBAA tags to global variables"));
static llvm::cl::opt<bool>
    enableDirect("direct-tbaa", llvm::cl::init(true), llvm::cl::Hidden,
                 llvm::cl::desc("Add TBAA tags to direct variables"));
// This is **known unsafe** (misscompare in spec2017/wrf_r). It should
// not be enabled by default.
// The code is kept so that these may be tried with new benchmarks to see if
// this is worth fixing in the future.
static llvm::cl::opt<bool> enableLocalAllocs(
    "local-alloc-tbaa", llvm::cl::init(false), llvm::cl::Hidden,
    llvm::cl::desc("Add TBAA tags to local allocations. UNSAFE."));

namespace {

/// Shared state per-module
class PassState {
public:
  /// memoised call to fir::AliasAnalysis::getSource
  inline const fir::AliasAnalysis::Source &getSource(mlir::Value value) {
    if (!analysisCache.contains(value))
      analysisCache.insert({value, analysis.getSource(value)});
    return analysisCache[value];
  }

  /// get the per-function TBAATree for this function
  inline const fir::TBAATree &getFuncTree(mlir::func::FuncOp func) {
    return forrest[func];
  }

private:
  fir::AliasAnalysis analysis;
  llvm::DenseMap<mlir::Value, fir::AliasAnalysis::Source> analysisCache;
  fir::TBAAForrest forrest;
};

class AddAliasTagsPass : public fir::impl::AddAliasTagsBase<AddAliasTagsPass> {
public:
  void runOnOperation() override;

private:
  /// The real workhorse of the pass. This is a runOnOperation() which
  /// operates on fir::FirAliasTagOpInterface, using some extra state
  void runOnAliasInterface(fir::FirAliasTagOpInterface op, PassState &state);
};

} // namespace

static fir::DeclareOp getDeclareOp(mlir::Value arg) {
  for (mlir::Operation *use : arg.getUsers())
    if (fir::DeclareOp declare = mlir::dyn_cast<fir::DeclareOp>(use))
      return declare;
  return nullptr;
}

/// Get the name of a function argument using the "fir.bindc_name" attribute,
/// or ""
static std::string getFuncArgName(mlir::Value arg) {
  // first try getting the name from the hlfir.declare
  if (fir::DeclareOp declare = getDeclareOp(arg))
    return declare.getUniqName().str();

  // get from attribute on function argument
  // always succeeds because arg is a function argument
  mlir::BlockArgument blockArg = mlir::cast<mlir::BlockArgument>(arg);
  assert(blockArg.getOwner() && blockArg.getOwner()->isEntryBlock() &&
         "arg is a function argument");
  mlir::FunctionOpInterface func = mlir::dyn_cast<mlir::FunctionOpInterface>(
      blockArg.getOwner()->getParentOp());
  mlir::StringAttr attr = func.getArgAttrOfType<mlir::StringAttr>(
      blockArg.getArgNumber(), "fir.bindc_name");
  if (!attr)
    return "";
  return attr.str();
}

void AddAliasTagsPass::runOnAliasInterface(fir::FirAliasTagOpInterface op,
                                           PassState &state) {
  mlir::func::FuncOp func = op->getParentOfType<mlir::func::FuncOp>();
  if (!func)
    return;

  llvm::SmallVector<mlir::Value> accessedOperands = op.getAccessedOperands();
  assert(accessedOperands.size() == 1 &&
         "load and store only access one address");
  mlir::Value memref = accessedOperands.front();

  // skip boxes. These get an "any descriptor access" tag in TBAABuilder
  // (CodeGen). I didn't see any speedup from giving each box a separate TBAA
  // type.
  if (mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(memref.getType())))
    return;
  LLVM_DEBUG(llvm::dbgs() << "Analysing " << op << "\n");

  const fir::AliasAnalysis::Source &source = state.getSource(memref);
  if (source.isTargetOrPointer()) {
    LLVM_DEBUG(llvm::dbgs().indent(2) << "Skipping TARGET/POINTER\n");
    // These will get an "any data access" tag in TBAABuilder (CodeGen): causing
    // them to "MayAlias" with all non-box accesses
    return;
  }

  mlir::LLVM::TBAATagAttr tag;
  // TBAA for dummy arguments
  if (enableDummyArgs &&
      source.kind == fir::AliasAnalysis::SourceKind::Argument) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "Found reference to dummy argument at " << *op << "\n");
    std::string name = getFuncArgName(source.u.get<mlir::Value>());
    if (!name.empty())
      tag = state.getFuncTree(func).dummyArgDataTree.getTag(name);
    else
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "WARN: couldn't find a name for dummy argument " << *op
                 << "\n");

    // TBAA for global variables
  } else if (enableGlobals &&
             source.kind == fir::AliasAnalysis::SourceKind::Global) {
    mlir::SymbolRefAttr glbl = source.u.get<mlir::SymbolRefAttr>();
    const char *name = glbl.getRootReference().data();
    LLVM_DEBUG(llvm::dbgs().indent(2) << "Found reference to global " << name
                                      << " at " << *op << "\n");
    tag = state.getFuncTree(func).globalDataTree.getTag(name);

    // TBAA for SourceKind::Direct
  } else if (enableDirect &&
             source.kind == fir::AliasAnalysis::SourceKind::Direct) {
    if (source.u.is<mlir::SymbolRefAttr>()) {
      mlir::SymbolRefAttr glbl = source.u.get<mlir::SymbolRefAttr>();
      const char *name = glbl.getRootReference().data();
      LLVM_DEBUG(llvm::dbgs().indent(2) << "Found reference to direct " << name
                                        << " at " << *op << "\n");
      tag = state.getFuncTree(func).directDataTree.getTag(name);
    } else {
      // SourceKind::Direct is likely to be extended to cases which are not a
      // SymbolRefAttr in the future
      LLVM_DEBUG(llvm::dbgs().indent(2) << "Can't get name for direct "
                                        << source << " at " << *op << "\n");
    }

    // TBAA for local allocations
  } else if (enableLocalAllocs &&
             source.kind == fir::AliasAnalysis::SourceKind::Allocate) {
    std::optional<llvm::StringRef> name;
    mlir::Operation *sourceOp = source.u.get<mlir::Value>().getDefiningOp();
    if (auto alloc = mlir::dyn_cast_or_null<fir::AllocaOp>(sourceOp))
      name = alloc.getUniqName();
    else if (auto alloc = mlir::dyn_cast_or_null<fir::AllocMemOp>(sourceOp))
      name = alloc.getUniqName();
    if (name) {
      LLVM_DEBUG(llvm::dbgs().indent(2) << "Found reference to allocation "
                                        << name << " at " << *op << "\n");
      tag = state.getFuncTree(func).allocatedDataTree.getTag(*name);
    } else {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "WARN: couldn't find a name for allocation " << *op
                 << "\n");
    }
  } else {
    if (source.kind != fir::AliasAnalysis::SourceKind::Argument &&
        source.kind != fir::AliasAnalysis::SourceKind::Allocate &&
        source.kind != fir::AliasAnalysis::SourceKind::Global)
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "WARN: unsupported value: " << source << "\n");
  }

  if (tag)
    op.setTBAATags(mlir::ArrayAttr::get(&getContext(), tag));
}

void AddAliasTagsPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "=== Begin " DEBUG_TYPE " ===\n");

  // MLIR forbids storing state in a pass because different instances might be
  // used in different threads.
  // Instead this pass stores state per mlir::ModuleOp (which is what MLIR
  // thinks the pass operates on), then the real work of the pass is done in
  // runOnAliasInterface
  PassState state;

  mlir::ModuleOp mod = getOperation();
  mod.walk(
      [&](fir::FirAliasTagOpInterface op) { runOnAliasInterface(op, state); });

  LLVM_DEBUG(llvm::dbgs() << "=== End " DEBUG_TYPE " ===\n");
}

std::unique_ptr<mlir::Pass> fir::createAliasTagsPass() {
  return std::make_unique<AddAliasTagsPass>();
}
