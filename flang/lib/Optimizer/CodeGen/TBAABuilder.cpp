//===-- TBAABuilder.cpp -- TBAA builder definitions -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "TBAABuilder.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-tbaa-builder"

using namespace mlir;
using namespace mlir::LLVM;

static llvm::cl::opt<bool> disableTBAA(
    "disable-tbaa",
    llvm::cl::desc("disable attaching TBAA tags to memory accessing operations "
                   "to override default Flang behavior"),
    llvm::cl::init(false));

// tagAttachmentLimit is a debugging option that allows limiting
// the number of TBAA access tag attributes attached to operations.
// It is set to kTagAttachmentUnlimited by default denoting "no limit".
static constexpr unsigned kTagAttachmentUnlimited =
    std::numeric_limits<unsigned>::max();
static llvm::cl::opt<unsigned>
    tagAttachmentLimit("tbaa-attach-tag-max", llvm::cl::desc(""),
                       llvm::cl::init(kTagAttachmentUnlimited));

namespace fir {
std::string TBAABuilder::getNewTBAANodeName(llvm::StringRef basename) {
  return (llvm::Twine(basename) + llvm::Twine('_') +
          llvm::Twine(tbaaNodeCounter++))
      .str();
}

TBAABuilder::TBAABuilder(mlir::ModuleOp module, bool applyTBAA)
    : enableTBAA(applyTBAA && !disableTBAA) {
  if (!enableTBAA)
    return;

  // In the usual Flang compilation flow, FIRToLLVMPass is run once,
  // and the MetadataOp holding TBAA operations is created at the beginning
  // of the pass. With tools like tco it is possible to invoke
  // FIRToLLVMPass on already converted MLIR, so the MetadataOp
  // already exists and creating a new one with the same name would
  // be incorrect. If the TBAA MetadataOp is already present,
  // we just disable all TBAABuilder actions (e.g. attachTBAATag()
  // is a no-op).
  if (llvm::any_of(
          module.getBodyRegion().getOps<LLVM::MetadataOp>(),
          [&](auto metaOp) { return metaOp.getSymName() == tbaaMetaOpName; })) {
    enableTBAA = false;
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Creating TBAA MetadataOp for module '"
                          << module.getName().value_or("<unknown>") << "'\n");

  // Create TBAA MetadataOp with the root and basic type descriptors.
  Location loc = module.getLoc();
  MLIRContext *context = module.getContext();
  OpBuilder builder(module.getBody(), module.getBody()->end());
  tbaaMetaOp = builder.create<MetadataOp>(loc, tbaaMetaOpName);
  builder.setInsertionPointToStart(&tbaaMetaOp.getBody().front());

  // Root node.
  auto rootOp = builder.create<TBAARootMetadataOp>(
      loc, getNewTBAANodeName(kRootSymBasename), flangTBAARootId);
  flangTBAARoot = FlatSymbolRefAttr::get(rootOp);

  // Any access node.
  auto anyAccessOp = builder.create<TBAATypeDescriptorOp>(
      loc, getNewTBAANodeName(kTypeDescSymBasename),
      StringAttr::get(context, anyAccessTypeDescId),
      ArrayAttr::get(context, flangTBAARoot), ArrayRef<int64_t>{0});
  anyAccessTypeDesc = FlatSymbolRefAttr::get(anyAccessOp);

  // Any data access node.
  auto anyDataAccessOp = builder.create<TBAATypeDescriptorOp>(
      loc, getNewTBAANodeName(kTypeDescSymBasename),
      StringAttr::get(context, anyDataAccessTypeDescId),
      ArrayAttr::get(context, anyAccessTypeDesc), ArrayRef<int64_t>{0});
  anyDataAccessTypeDesc = FlatSymbolRefAttr::get(anyDataAccessOp);

  // Box member access node.
  auto boxMemberOp = builder.create<TBAATypeDescriptorOp>(
      loc, getNewTBAANodeName(kTypeDescSymBasename),
      StringAttr::get(context, boxMemberTypeDescId),
      ArrayAttr::get(context, anyAccessTypeDesc), ArrayRef<int64_t>{0});
  boxMemberTypeDesc = FlatSymbolRefAttr::get(boxMemberOp);
}

SymbolRefAttr TBAABuilder::getAccessTag(SymbolRefAttr baseTypeDesc,
                                        SymbolRefAttr accessTypeDesc,
                                        int64_t offset) {
  SymbolRefAttr &tag = tagsMap[{baseTypeDesc, accessTypeDesc, offset}];
  if (tag)
    return tag;

  // Initialize new tag.
  Location loc = tbaaMetaOp.getLoc();
  OpBuilder builder(&tbaaMetaOp.getBody().back(),
                    tbaaMetaOp.getBody().back().end());
  auto tagOp = builder.create<TBAATagOp>(
      loc, getNewTBAANodeName(kTagSymBasename), baseTypeDesc.getLeafReference(),
      accessTypeDesc.getLeafReference(), offset);
  // TBAATagOp symbols must be referenced by their fully qualified
  // names, so create a path to TBAATagOp symbol.
  StringAttr metaOpName = SymbolTable::getSymbolName(tbaaMetaOp);
  tag = SymbolRefAttr::get(builder.getContext(), metaOpName,
                           FlatSymbolRefAttr::get(tagOp));
  return tag;
}

SymbolRefAttr TBAABuilder::getAnyBoxAccessTag() {
  return getAccessTag(boxMemberTypeDesc, boxMemberTypeDesc, /*offset=*/0);
}

SymbolRefAttr TBAABuilder::getBoxAccessTag(Type baseFIRType, Type accessFIRType,
                                           GEPOp gep) {
  return getAnyBoxAccessTag();
}

SymbolRefAttr TBAABuilder::getAnyDataAccessTag() {
  return getAccessTag(anyDataAccessTypeDesc, anyDataAccessTypeDesc,
                      /*offset=*/0);
}

SymbolRefAttr TBAABuilder::getDataAccessTag(Type baseFIRType,
                                            Type accessFIRType, GEPOp gep) {
  return getAnyDataAccessTag();
}

void TBAABuilder::attachTBAATag(Operation *op, Type baseFIRType,
                                Type accessFIRType, GEPOp gep) {
  if (!enableTBAA)
    return;

  ++tagAttachmentCounter;
  if (tagAttachmentLimit != kTagAttachmentUnlimited &&
      tagAttachmentCounter > tagAttachmentLimit)
    return;

  LLVM_DEBUG(llvm::dbgs() << "Attaching TBAA tag #" << tagAttachmentCounter
                          << "\n");

  SymbolRefAttr tbaaTagSym;
  if (baseFIRType.isa<fir::BaseBoxType>())
    tbaaTagSym = getBoxAccessTag(baseFIRType, accessFIRType, gep);
  else
    tbaaTagSym = getDataAccessTag(baseFIRType, accessFIRType, gep);

  if (tbaaTagSym)
    op->setAttr(LLVMDialect::getTBAAAttrName(),
                ArrayAttr::get(op->getContext(), tbaaTagSym));
}

} // namespace fir
