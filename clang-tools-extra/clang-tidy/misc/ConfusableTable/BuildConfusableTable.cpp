//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
struct ConfusableEntry {
  UTF32 CodePoint;
  std::string Replacement;
};
} // namespace

int main(int argc, char *argv[]) {
  auto ErrorOrBuffer = MemoryBuffer::getFile(argv[1], true);
  if (!ErrorOrBuffer)
    return 1;
  std::unique_ptr<MemoryBuffer> Buffer = std::move(ErrorOrBuffer.get());
  StringRef Content = Buffer->getBuffer();
  Content = Content.drop_until([](char C) { return C == '#'; });
  SmallVector<StringRef> Lines;
  SplitString(Content, Lines, "\r\n");

  std::vector<ConfusableEntry> Entries;
  SmallVector<StringRef> Values;
  for (const StringRef Line : Lines) {
    if (Line.starts_with('#'))
      continue;

    Values.clear();
    Line.split(Values, ';');
    if (Values.size() < 2) {
      errs() << "Failed to parse: " << Line << "\n";
      return 2;
    }

    const StringRef From = Values[0].trim();
    llvm::UTF32 CodePoint = 0;
    From.getAsInteger(16, CodePoint);

    std::string Replacement;
    SmallVector<StringRef> ToN;
    Values[1].split(ToN, ' ', -1, false);
    for (const StringRef ToI : ToN) {
      llvm::UTF32 ToCodePoint = 0;
      ToI.trim().getAsInteger(16, ToCodePoint);
      char Encoded[UNI_MAX_UTF8_BYTES_PER_CODE_POINT];
      char *EncodedEnd = Encoded;
      if (!ConvertCodePointToUTF8(ToCodePoint, EncodedEnd)) {
        errs() << "Failed to encode code point: " << ToI << "\n";
        return 3;
      }
      Replacement.append(Encoded, EncodedEnd);
    }

    Entries.push_back({CodePoint, std::move(Replacement)});
  }
  llvm::sort(Entries,
             [](const ConfusableEntry &LHS, const ConfusableEntry &RHS) {
               return LHS.CodePoint < RHS.CodePoint;
             });

  StringMap<uint16_t> ReplacementOffsets;
  std::string ReplacementData;
  for (const ConfusableEntry &Entry : Entries) {
    if (ReplacementOffsets.contains(Entry.Replacement))
      continue;
    if (ReplacementData.size() + Entry.Replacement.size() >
        std::numeric_limits<uint16_t>::max()) {
      errs() << "Confusable replacement data exceeds 16-bit offsets\n";
      return 4;
    }
    ReplacementOffsets.try_emplace(
        Entry.Replacement, static_cast<uint16_t>(ReplacementData.size()));
    ReplacementData.append(Entry.Replacement);
  }

  std::error_code Ec;
  llvm::raw_fd_ostream Os(argv[2], Ec);

  static constexpr char HexDigits[] = "0123456789ABCDEF";
  Os << "constexpr char ConfusableReplacementData[] =\n";
  for (size_t I = 0; I < ReplacementData.size(); I += 32) {
    Os << "    \"";
    for (const unsigned char C : StringRef(ReplacementData).substr(I, 32))
      Os << "\\x" << HexDigits[C >> 4] << HexDigits[C & 0x0F];
    Os << "\"\n";
  }
  Os << "    ;\n\n";

  Os << "constexpr ConfusableEntry ConfusableEntries[] = {\n";
  for (const ConfusableEntry &Entry : Entries) {
    const auto It = ReplacementOffsets.find(Entry.Replacement);
    Os << "  {" << Entry.CodePoint << ", " << It->second << ", "
       << Entry.Replacement.size() << "},\n";
  }
  Os << "};\n";
  return 0;
}
