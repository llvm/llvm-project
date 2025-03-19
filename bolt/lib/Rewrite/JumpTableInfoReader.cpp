//===- bolt/Rewrite/JumpTableInfoReader.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Read .llvm_jump_table_info section and register jump tables.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/JumpTable.h"
#include "bolt/Rewrite/MetadataRewriter.h"
#include "bolt/Rewrite/MetadataRewriters.h"
#include "llvm/Support/DataExtractor.h"

using namespace llvm;
using namespace bolt;

namespace {
class JumpTableInfoReader final : public MetadataRewriter {

public:
  JumpTableInfoReader(StringRef Name, BinaryContext &BC)
      : MetadataRewriter(Name, BC) {}
  Error preDisasmInitializer() override;
};

Error JumpTableInfoReader::preDisasmInitializer() {
  if (!BC.isAArch64())
    return Error::success();

  ErrorOr<BinarySection &> ErrorOrJTInfoSection =
      BC.getUniqueSectionByName(".llvm_jump_table_info");
  if (std::error_code E = ErrorOrJTInfoSection.getError())
    return Error::success();
  BinarySection &JTInfoSection = *ErrorOrJTInfoSection;
  StringRef Buf = JTInfoSection.getContents();
  DataExtractor DE = DataExtractor(Buf, BC.AsmInfo->isLittleEndian(),
                                   BC.AsmInfo->getCodePointerSize());
  DataExtractor::Cursor Cursor(0);
  while (Cursor && !DE.eof(Cursor)) {
    const uint8_t Format = DE.getU8(Cursor);
    const uint64_t JTAddr = DE.getAddress(Cursor);
    const uint64_t JTBase = DE.getAddress(Cursor);
    const uint64_t JTLoad = DE.getAddress(Cursor);
    const uint64_t Branch = DE.getAddress(Cursor);
    const uint64_t NumEntries = DE.getULEB128(Cursor);

    JumpTable::JumpTableType Type = JumpTable::JTT_AARCH64_LAST;
    switch (Format) {
    case 2:
      Type = JumpTable::JTT_AARCH64_REL1;
      break;
    case 3:
      Type = JumpTable::JTT_AARCH64_REL2;
      break;
    case 4:
      Type = JumpTable::JTT_AARCH64_REL4;
      break;
    }

    if (Type == JumpTable::JTT_AARCH64_LAST) {
      errs() << "BOLT-WARNING: unknown jump table info type " << Format
                << " for jump table " << Twine::utohexstr(JTAddr) << '\n';
      continue;
    }

    BinaryFunction *BF = BC.getBinaryFunctionContainingAddress(Branch);
    if (!BF) {
      BC.errs() << "BOLT-WARNING: binary function not found for jump table "
                   "with address "
                << Twine::utohexstr(JTAddr) << " and branch "
                << Twine::utohexstr(Branch) << '\n';
      continue;
    }
    const MCSymbol *JTSym = BC.getOrCreateJumpTable(*BF, JTAddr, Type);
    assert(JTSym && "failed to create a jump table");
    JumpTable *JT = BC.getJumpTableContainingAddress(JTAddr);
    assert(JT && "internal error creating jump table");
    JT->BaseAddress = JTBase;
    JT->MemLocInstrAddress = JTLoad;
    JT->Entries.resize(NumEntries);
  }
  return Cursor.takeError();
}
} // namespace

std::unique_ptr<MetadataRewriter>
llvm::bolt::createJumpTableInfoReader(BinaryContext &BC) {
  return std::make_unique<JumpTableInfoReader>("jump-table-info-reader", BC);
}
