//===- UnwindInfoSection.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_UNWIND_INFO_H
#define LLD_MACHO_UNWIND_INFO_H

#include "MergedOutputSection.h"
#include "SyntheticSections.h"

#include "mach-o/compact_unwind_encoding.h"
#include "llvm/ADT/DenseMap.h"

#include <vector>

// In 2020, we mostly care about 64-bit targets: x86_64 and arm64
struct CompactUnwindEntry64 {
  uint64_t functionAddress;
  uint32_t functionLength;
  compact_unwind_encoding_t encoding;
  uint64_t personality;
  uint64_t lsda;
};

// FIXME(gkm): someday we might care about 32-bit targets: x86 & arm
struct CompactUnwindEntry32 {
  uint32_t functionAddress;
  uint32_t functionLength;
  compact_unwind_encoding_t encoding;
  uint32_t personality;
  uint32_t lsda;
};

namespace lld {
namespace macho {

class UnwindInfoSection : public SyntheticSection {
public:
  UnwindInfoSection();
  uint64_t getSize() const override { return unwindInfoSize; }
  bool isNeeded() const override;
  void finalize() override;
  void writeTo(uint8_t *buf) const override;
  void setCompactUnwindSection(MergedOutputSection *cuSection) {
    compactUnwindSection = cuSection;
  }

private:
  std::vector<std::pair<compact_unwind_encoding_t, size_t>> commonEncodings;
  std::vector<uint32_t> personalities;
  std::vector<unwind_info_section_header_lsda_index_entry> lsdaEntries;
  std::vector<CompactUnwindEntry64> cuVector;
  std::vector<const CompactUnwindEntry64 *> cuPtrVector;
  std::vector<std::vector<const CompactUnwindEntry64 *>::const_iterator>
      pageBounds;
  MergedOutputSection *compactUnwindSection = nullptr;
  uint64_t level2PagesOffset = 0;
  uint64_t unwindInfoSize = 0;
};

#define UNWIND_INFO_COMMON_ENCODINGS_MAX 127

#define UNWIND_INFO_SECOND_LEVEL_PAGE_SIZE 4096
#define UNWIND_INFO_REGULAR_SECOND_LEVEL_ENTRIES_MAX                           \
  ((UNWIND_INFO_SECOND_LEVEL_PAGE_SIZE -                                       \
    sizeof(unwind_info_regular_second_level_page_header)) /                    \
   sizeof(unwind_info_regular_second_level_entry))
#define UNWIND_INFO_COMPRESSED_SECOND_LEVEL_ENTRIES_MAX                        \
  ((UNWIND_INFO_SECOND_LEVEL_PAGE_SIZE -                                       \
    sizeof(unwind_info_compressed_second_level_page_header)) /                 \
   sizeof(uint32_t))

#define UNWIND_INFO_COMPRESSED_ENTRY_FUNC_OFFSET_BITS 24
#define UNWIND_INFO_COMPRESSED_ENTRY_FUNC_OFFSET_MASK                          \
  UNWIND_INFO_COMPRESSED_ENTRY_FUNC_OFFSET(~0)

} // namespace macho
} // namespace lld

#endif
