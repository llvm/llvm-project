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

#include "flang/Optimizer/CodeGen/TBAABuilder.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "llvm/ADT/TypeSwitch.h"
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

TBAABuilder::TBAABuilder(MLIRContext *context, bool applyTBAA)
    : enableTBAA(applyTBAA && !disableTBAA) {
  if (!enableTBAA)
    return;

  // Root node.
  flangTBAARoot =
      TBAARootAttr::get(context, StringAttr::get(context, flangTBAARootId));

  // Any access node.
  anyAccessTypeDesc = TBAATypeDescriptorAttr::get(
      context, anyAccessTypeDescId, TBAAMemberAttr::get(flangTBAARoot, 0));

  // Any data access node.
  anyDataAccessTypeDesc =
      TBAATypeDescriptorAttr::get(context, anyDataAccessTypeDescId,
                                  TBAAMemberAttr::get(anyAccessTypeDesc, 0));

  // Box member access node.
  boxMemberTypeDesc = TBAATypeDescriptorAttr::get(
      context, boxMemberTypeDescId, TBAAMemberAttr::get(anyAccessTypeDesc, 0));
}

TBAATagAttr TBAABuilder::getAccessTag(TBAATypeDescriptorAttr baseTypeDesc,
                                      TBAATypeDescriptorAttr accessTypeDesc,
                                      int64_t offset) {
  TBAATagAttr &tag = tagsMap[{baseTypeDesc, accessTypeDesc, offset}];
  if (tag)
    return tag;

  // Initialize new tag.
  tag = TBAATagAttr::get(baseTypeDesc, accessTypeDesc, offset);
  return tag;
}

TBAATagAttr TBAABuilder::getAnyBoxAccessTag() {
  return getAccessTag(boxMemberTypeDesc, boxMemberTypeDesc, /*offset=*/0);
}

TBAATagAttr TBAABuilder::getBoxAccessTag(Type baseFIRType, Type accessFIRType,
                                         GEPOp gep) {
  return getAnyBoxAccessTag();
}

TBAATagAttr TBAABuilder::getAnyDataAccessTag() {
  return getAccessTag(anyDataAccessTypeDesc, anyDataAccessTypeDesc,
                      /*offset=*/0);
}

TBAATagAttr TBAABuilder::getDataAccessTag(Type baseFIRType, Type accessFIRType,
                                          GEPOp gep) {
  return getAnyDataAccessTag();
}

TBAATagAttr TBAABuilder::getAnyAccessTag() {
  return getAccessTag(anyAccessTypeDesc, anyAccessTypeDesc, /*offset=*/0);
}

void TBAABuilder::attachTBAATag(AliasAnalysisOpInterface op, Type baseFIRType,
                                Type accessFIRType, GEPOp gep) {
  if (!enableTBAA)
    return;

  ++tagAttachmentCounter;
  if (tagAttachmentLimit != kTagAttachmentUnlimited &&
      tagAttachmentCounter > tagAttachmentLimit)
    return;

  LLVM_DEBUG(llvm::dbgs() << "Attaching TBAA tag #" << tagAttachmentCounter
                          << "\n");

  TBAATagAttr tbaaTagSym;
  if (fir::isRecordWithDescriptorMember(baseFIRType)) {
    // A memory access that addresses an aggregate that contains
    // a mix of data members and descriptor members may alias
    // with both data and descriptor accesses.
    // Conservatively set any-access tag if there is any descriptor member.
    tbaaTagSym = getAnyAccessTag();
  } else if (baseFIRType.isa<fir::BaseBoxType>()) {
    tbaaTagSym = getBoxAccessTag(baseFIRType, accessFIRType, gep);
  } else {
    tbaaTagSym = getDataAccessTag(baseFIRType, accessFIRType, gep);
  }

  if (!tbaaTagSym)
    return;

  op.setTBAATags(ArrayAttr::get(op->getContext(), tbaaTagSym));
}

} // namespace fir
