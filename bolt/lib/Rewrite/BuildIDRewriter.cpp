//===- bolt/Rewrite/BuildIDRewriter.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Read and update build ID stored in ELF note section.
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/MetadataRewriter.h"
#include "bolt/Rewrite/MetadataRewriters.h"
#include "llvm/Support/Errc.h"

using namespace llvm;
using namespace bolt;

namespace {

/// The build-id is typically a stream of 20 bytes. Return these bytes in
/// printable hexadecimal form.
std::string getPrintableBuildID(StringRef BuildID) {
  std::string Str;
  raw_string_ostream OS(Str);
  for (const char &Char : BuildID)
    OS << format("%.2x", static_cast<unsigned char>(Char));

  return OS.str();
}

class BuildIDRewriter final : public MetadataRewriter {

  /// Information about binary build ID.
  ErrorOr<BinarySection &> BuildIDSection{std::errc::bad_address};
  StringRef BuildID;
  std::optional<uint64_t> BuildIDOffset;
  std::optional<uint64_t> BuildIDSize;

public:
  BuildIDRewriter(StringRef Name, BinaryContext &BC)
      : MetadataRewriter(Name, BC) {}

  Error sectionInitializer() override;

  Error postEmitFinalizer() override;
};

Error BuildIDRewriter::sectionInitializer() {
  // Typically, build ID will reside in .note.gnu.build-id section. However,
  // a linker script can change the section name and such is the case with
  // the Linux kernel. Hence, we iterate over all note sections.
  for (BinarySection &NoteSection : BC.sections()) {
    if (!NoteSection.isNote())
      continue;

    StringRef Buf = NoteSection.getContents();
    DataExtractor DE = DataExtractor(Buf, BC.AsmInfo->isLittleEndian(),
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
        return createStringError(errc::executable_format_error,
                                 "out of bounds while reading note section: %s",
                                 toString(Cursor.takeError()).c_str());

      if (Type == ELF::NT_GNU_BUILD_ID && Name.starts_with("GNU") && DescSz) {
        BuildIDSection = NoteSection;
        BuildID = Desc;
        BC.setFileBuildID(getPrintableBuildID(Desc));
        BuildIDOffset = DescOffset;
        BuildIDSize = DescSz;

        return Error::success();
      }
    }
  }

  return Error::success();
}

Error BuildIDRewriter::postEmitFinalizer() {
  if (!BuildIDSection || !BuildIDOffset)
    return Error::success();

  const uint8_t LastByte = BuildID[BuildID.size() - 1];
  SmallVector<char, 1> Patch = {static_cast<char>(LastByte ^ 1)};
  BuildIDSection->addPatch(*BuildIDOffset + BuildID.size() - 1, Patch);
  BC.outs() << "BOLT-INFO: patched build-id (flipped last bit)\n";

  return Error::success();
}
} // namespace

std::unique_ptr<MetadataRewriter>
llvm::bolt::createBuildIDRewriter(BinaryContext &BC) {
  return std::make_unique<BuildIDRewriter>("build-id-rewriter", BC);
}
