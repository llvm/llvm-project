//===- bolt/Rewrite/GNUPropertyRewriter.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Read the .note.gnu.property section.
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/MetadataRewriter.h"
#include "bolt/Rewrite/MetadataRewriters.h"
#include "llvm/Support/Errc.h"

using namespace llvm;
using namespace bolt;

namespace {

class GNUPropertyRewriter final : public MetadataRewriter {

  Expected<uint32_t> decodeGNUPropertyNote(StringRef Desc);

public:
  GNUPropertyRewriter(StringRef Name, BinaryContext &BC)
      : MetadataRewriter(Name, BC) {}

  Error sectionInitializer() override;
};

Error GNUPropertyRewriter::sectionInitializer() {

  ErrorOr<BinarySection &> Sec =
      BC.getUniqueSectionByName(".note.gnu.property");
  if (!Sec)
    return Error::success();

  // Accumulate feature bits
  uint32_t FeaturesAcc = 0;

  StringRef Buf = Sec->getContents();
  DataExtractor DE(Buf, BC.AsmInfo->isLittleEndian(),
                   BC.AsmInfo->getCodePointerSize());
  DataExtractor::Cursor Cursor(0);
  while (Cursor && !DE.eof(Cursor)) {
    const uint32_t NameSz = DE.getU32(Cursor);
    const uint32_t DescSz = DE.getU32(Cursor);
    const uint32_t Type = DE.getU32(Cursor);

    StringRef Name =
        NameSz ? Buf.slice(Cursor.tell(), Cursor.tell() + NameSz) : "<empty>";
    Cursor.seek(alignTo(Cursor.tell() + NameSz, 4));

    const uint64_t DescOffset = Cursor.tell();
    StringRef Desc =
        DescSz ? Buf.slice(DescOffset, DescOffset + DescSz) : "<empty>";
    Cursor.seek(alignTo(DescOffset + DescSz, 4));
    if (!Cursor)
      return createStringError(
          errc::executable_format_error,
          "out of bounds while reading .note.gnu.property section: %s",
          toString(Cursor.takeError()).c_str());

    if (Type == ELF::NT_GNU_PROPERTY_TYPE_0 && Name.starts_with("GNU") &&
        DescSz) {
      auto Features = decodeGNUPropertyNote(Desc);
      if (!Features)
        return Features.takeError();
      FeaturesAcc |= *Features;
    }
  }

  if (BC.isAArch64()) {
    BC.setUsesBTI(FeaturesAcc & llvm::ELF::GNU_PROPERTY_AARCH64_FEATURE_1_BTI);
    if (BC.usesBTI())
      BC.outs() << "BOLT-WARNING: binary is using BTI. Optimized binary may be "
                   "corrupted\n";
  }

  return Error::success();
}

/// \p Desc contains an array of property descriptors. Each member has the
/// following structure:
/// typedef struct {
///   Elf_Word pr_type;
///   Elf_Word pr_datasz;
///   unsigned char pr_data[PR_DATASZ];
///   unsigned char pr_padding[PR_PADDING];
/// } Elf_Prop;
///
/// As there is no guarantee that the features are encoded in which element of
/// the array, we have to read all, and OR together the result.
Expected<uint32_t> GNUPropertyRewriter::decodeGNUPropertyNote(StringRef Desc) {
  DataExtractor DE(Desc, BC.AsmInfo->isLittleEndian(),
                   BC.AsmInfo->getCodePointerSize());
  DataExtractor::Cursor Cursor(0);
  const uint32_t Align = DE.getAddressSize();

  std::optional<uint32_t> Features = 0;
  while (Cursor && !DE.eof(Cursor)) {
    const uint32_t PrType = DE.getU32(Cursor);
    const uint32_t PrDataSz = DE.getU32(Cursor);

    const uint64_t PrDataStart = Cursor.tell();
    const uint64_t PrDataEnd = PrDataStart + PrDataSz;
    Cursor.seek(PrDataEnd);
    if (!Cursor)
      return createStringError(
          errc::executable_format_error,
          "out of bounds while reading .note.gnu.property section: %s",
          toString(Cursor.takeError()).c_str());

    if (PrType == llvm::ELF::GNU_PROPERTY_AARCH64_FEATURE_1_AND) {
      if (PrDataSz != 4) {
        return createStringError(
            errc::executable_format_error,
            "Property descriptor size has to be 4 bytes on AArch64\n");
      }
      DataExtractor::Cursor Tmp(PrDataStart);
      // PrDataSz = 4 -> PrData is uint32_t
      const uint32_t FeaturesItem = DE.getU32(Tmp);
      if (!Tmp)
        return createStringError(
            errc::executable_format_error,
            "failed to read property from .note.gnu.property section: %s",
            toString(Tmp.takeError()).c_str());
      Features = Features ? (*Features | FeaturesItem) : FeaturesItem;
    }

    Cursor.seek(alignTo(PrDataEnd, Align));
    if (!Cursor)
      return createStringError(errc::executable_format_error,
                               "out of bounds while reading property array in "
                               ".note.gnu.property section: %s",
                               toString(Cursor.takeError()).c_str());
  }
  return Features.value_or(0u);
}
} // namespace

std::unique_ptr<MetadataRewriter>
llvm::bolt::createGNUPropertyRewriter(BinaryContext &BC) {
  return std::make_unique<GNUPropertyRewriter>("gnu-property-rewriter", BC);
}
