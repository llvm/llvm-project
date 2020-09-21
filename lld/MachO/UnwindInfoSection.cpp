//===- UnwindInfoSection.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnwindInfoSection.h"
#include "Config.h"
#include "InputSection.h"
#include "MergedOutputSection.h"
#include "OutputSection.h"
#include "OutputSegment.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"

#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/MachO.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace lld;
using namespace lld::macho;

// Compact Unwind format is a Mach-O evolution of DWARF Unwind that
// optimizes space and exception-time lookup.  Most DWARF unwind
// entries can be replaced with Compact Unwind entries, but the ones
// that cannot are retained in DWARF form.
//
// This comment will address macro-level organization of the pre-link
// and post-link compact unwind tables. For micro-level organization
// pertaining to the bitfield layout of the 32-bit compact unwind
// entries, see libunwind/include/mach-o/compact_unwind_encoding.h
//
// Important clarifying factoids:
//
// * __LD,__compact_unwind is the compact unwind format for compiler
// output and linker input. It is never a final output. It could be
// an intermediate output with the `-r` option which retains relocs.
//
// * __TEXT,__unwind_info is the compact unwind format for final
// linker output. It is never an input.
//
// * __TEXT,__eh_frame is the DWARF format for both linker input and output.
//
// * __TEXT,__unwind_info entries are divided into 4 KiB pages (2nd
// level) by ascending address, and the pages are referenced by an
// index (1st level) in the section header.
//
// * Following the headers in __TEXT,__unwind_info, the bulk of the
// section contains a vector of compact unwind entries
// `{functionOffset, encoding}` sorted by ascending `functionOffset`.
// Adjacent entries with the same encoding can be folded to great
// advantage, achieving a 3-order-of-magnitude reduction in the
// number of entries.
//
// * The __TEXT,__unwind_info format can accommodate up to 127 unique
// encodings for the space-efficient compressed format. In practice,
// fewer than a dozen unique encodings are used by C++ programs of
// all sizes. Therefore, we don't even bother implementing the regular
// non-compressed format. Time will tell if anyone in the field ever
// overflows the 127-encodings limit.

// TODO(gkm): prune __eh_frame entries superseded by __unwind_info
// TODO(gkm): how do we align the 2nd-level pages?

UnwindInfoSection::UnwindInfoSection()
    : SyntheticSection(segment_names::text, section_names::unwindInfo) {
  align = WordSize; // TODO(gkm): make this 4 KiB ?
}

bool UnwindInfoSection::isNeeded() const {
  return (compactUnwindSection != nullptr);
}

// Scan the __LD,__compact_unwind entries and compute the space needs of
// __TEXT,__unwind_info and __TEXT,__eh_frame

void UnwindInfoSection::finalize() {
  if (compactUnwindSection == nullptr)
    return;

  // At this point, the address space for __TEXT,__text has been
  // assigned, so we can relocate the __LD,__compact_unwind entries
  // into a temporary buffer. Relocation is necessary in order to sort
  // the CU entries by function address. Sorting is necessary so that
  // we can fold adjacent CU entries with identical
  // encoding+personality+lsda. Folding is necessary because it reduces
  // the number of CU entries by as much as 3 orders of magnitude!
  compactUnwindSection->finalize();
  assert(compactUnwindSection->getSize() % sizeof(CompactUnwindEntry64) == 0);
  size_t cuCount =
      compactUnwindSection->getSize() / sizeof(CompactUnwindEntry64);
  cuVector.resize(cuCount);
  // Relocate all __LD,__compact_unwind entries
  compactUnwindSection->writeTo(reinterpret_cast<uint8_t *>(cuVector.data()));

  // Rather than sort & fold the 32-byte entries directly, we create a
  // vector of pointers to entries and sort & fold that instead.
  cuPtrVector.reserve(cuCount);
  for (const auto &cuEntry : cuVector)
    cuPtrVector.emplace_back(&cuEntry);
  std::sort(cuPtrVector.begin(), cuPtrVector.end(),
            [](const CompactUnwindEntry64 *a, const CompactUnwindEntry64 *b) {
              return a->functionAddress < b->functionAddress;
            });

  // Fold adjacent entries with matching encoding+personality+lsda
  // We use three iterators on the same cuPtrVector to fold in-situ:
  // (1) `foldBegin` is the first of a potential sequence of matching entries
  // (2) `foldEnd` is the first non-matching entry after `foldBegin`.
  // The semi-open interval [ foldBegin .. foldEnd ) contains a range
  // entries that can be folded into a single entry and written to ...
  // (3) `foldWrite`
  auto foldWrite = cuPtrVector.begin();
  for (auto foldBegin = cuPtrVector.begin(); foldBegin < cuPtrVector.end();) {
    auto foldEnd = foldBegin;
    while (++foldEnd < cuPtrVector.end() &&
           (*foldBegin)->encoding == (*foldEnd)->encoding &&
           (*foldBegin)->personality == (*foldEnd)->personality &&
           (*foldBegin)->lsda == (*foldEnd)->lsda)
      ;
    *foldWrite++ = *foldBegin;
    foldBegin = foldEnd;
  }
  cuPtrVector.erase(foldWrite, cuPtrVector.end());

  // Count frequencies of the folded encodings
  llvm::DenseMap<compact_unwind_encoding_t, size_t> encodingFrequencies;
  for (auto cuPtrEntry : cuPtrVector)
    encodingFrequencies[cuPtrEntry->encoding]++;
  if (encodingFrequencies.size() > UNWIND_INFO_COMMON_ENCODINGS_MAX)
    error("TODO(gkm): handle common encodings table overflow");

  // Make a table of encodings, sorted by descending frequency
  for (const auto &frequency : encodingFrequencies)
    commonEncodings.emplace_back(frequency);
  std::sort(commonEncodings.begin(), commonEncodings.end(),
            [](const std::pair<compact_unwind_encoding_t, size_t> &a,
               const std::pair<compact_unwind_encoding_t, size_t> &b) {
              if (a.second == b.second)
                // When frequencies match, secondarily sort on encoding
                // to maintain parity with validate-unwind-info.py
                return a.first > b.first;
              return a.second > b.second;
            });

  // Split folded encodings into pages, limited by capacity of a page
  // and the 24-bit range of function offset
  //
  // Record the page splits as a vector of iterators on cuPtrVector
  // such that successive elements form a semi-open interval. E.g.,
  // page X's bounds are thus: [ pageBounds[X] .. pageBounds[X+1] )
  //
  // Note that pageBounds.size() is one greater than the number of
  // pages, and pageBounds.back() holds the sentinel cuPtrVector.cend()
  pageBounds.push_back(cuPtrVector.cbegin());
  // TODO(gkm): cut 1st page entries short to accommodate section headers ???
  CompactUnwindEntry64 cuEntryKey;
  for (size_t i = 0;;) {
    // Limit the search to entries that can fit within a 4 KiB page.
    const auto pageBegin = pageBounds[0] + i;
    const auto pageMax =
        pageBounds[0] +
        std::min(i + UNWIND_INFO_COMPRESSED_SECOND_LEVEL_ENTRIES_MAX,
                 cuPtrVector.size());
    // Exclude entries with functionOffset that would overflow 24 bits
    cuEntryKey.functionAddress = (*pageBegin)->functionAddress +
                                 UNWIND_INFO_COMPRESSED_ENTRY_FUNC_OFFSET_MASK;
    const auto pageBreak = std::lower_bound(
        pageBegin, pageMax, &cuEntryKey,
        [](const CompactUnwindEntry64 *a, const CompactUnwindEntry64 *b) {
          return a->functionAddress < b->functionAddress;
        });
    pageBounds.push_back(pageBreak);
    if (pageBreak == cuPtrVector.cend())
      break;
    i = pageBreak - cuPtrVector.cbegin();
  }

  // compute size of __TEXT,__unwind_info section
  level2PagesOffset =
      sizeof(unwind_info_section_header) +
      commonEncodings.size() * sizeof(uint32_t) +
      personalities.size() * sizeof(uint32_t) +
      pageBounds.size() * sizeof(unwind_info_section_header_index_entry) +
      lsdaEntries.size() * sizeof(unwind_info_section_header_lsda_index_entry);
  unwindInfoSize = level2PagesOffset +
                   (pageBounds.size() - 1) *
                       sizeof(unwind_info_compressed_second_level_page_header) +
                   cuPtrVector.size() * sizeof(uint32_t);
}

// All inputs are relocated and output adddresses are known, so write!

void UnwindInfoSection::writeTo(uint8_t *buf) const {
  // section header
  auto *uip = reinterpret_cast<unwind_info_section_header *>(buf);
  uip->version = 1;
  uip->commonEncodingsArraySectionOffset = sizeof(unwind_info_section_header);
  uip->commonEncodingsArrayCount = commonEncodings.size();
  uip->personalityArraySectionOffset =
      uip->commonEncodingsArraySectionOffset +
      (uip->commonEncodingsArrayCount * sizeof(uint32_t));
  uip->personalityArrayCount = personalities.size();
  uip->indexSectionOffset = uip->personalityArraySectionOffset +
                            (uip->personalityArrayCount * sizeof(uint32_t));
  uip->indexCount = pageBounds.size();

  // Common encodings
  auto *i32p = reinterpret_cast<uint32_t *>(&uip[1]);
  for (const auto &encoding : commonEncodings)
    *i32p++ = encoding.first;

  // Personalities
  for (const auto &personality : personalities)
    *i32p++ = personality;

  // Level-1 index
  uint32_t lsdaOffset =
      uip->indexSectionOffset +
      uip->indexCount * sizeof(unwind_info_section_header_index_entry);
  uint64_t l2PagesOffset = level2PagesOffset;
  auto *iep = reinterpret_cast<unwind_info_section_header_index_entry *>(i32p);
  for (size_t i = 0; i < pageBounds.size() - 1; i++) {
    iep->functionOffset = (*pageBounds[i])->functionAddress;
    iep->secondLevelPagesSectionOffset = l2PagesOffset;
    iep->lsdaIndexArraySectionOffset = lsdaOffset;
    iep++;
    // TODO(gkm): pad to 4 KiB page boundary ???
    size_t entryCount = pageBounds[i + 1] - pageBounds[i];
    uint64_t pageSize = sizeof(unwind_info_section_header_index_entry) +
                        entryCount * sizeof(uint32_t);
    l2PagesOffset += pageSize;
  }
  // Level-1 sentinel
  const CompactUnwindEntry64 &cuEnd = cuVector.back();
  iep->functionOffset = cuEnd.functionAddress + cuEnd.functionLength;
  iep->secondLevelPagesSectionOffset = 0;
  iep->lsdaIndexArraySectionOffset = lsdaOffset;
  iep++;

  // LSDAs
  auto *lep =
      reinterpret_cast<unwind_info_section_header_lsda_index_entry *>(iep);
  for (const auto &lsda : lsdaEntries) {
    lep->functionOffset = lsda.functionOffset;
    lep->lsdaOffset = lsda.lsdaOffset;
  }

  // create map from encoding to common-encoding-table index compact
  // encoding entries use 7 bits to index the common-encoding table
  size_t i = 0;
  llvm::DenseMap<compact_unwind_encoding_t, size_t> commonEncodingIndexes;
  for (const auto &encoding : commonEncodings)
    commonEncodingIndexes[encoding.first] = i++;

  // Level-2 pages
  auto *p2p =
      reinterpret_cast<unwind_info_compressed_second_level_page_header *>(lep);
  for (size_t i = 0; i < pageBounds.size() - 1; i++) {
    p2p->kind = UNWIND_SECOND_LEVEL_COMPRESSED;
    p2p->entryPageOffset =
        sizeof(unwind_info_compressed_second_level_page_header);
    p2p->entryCount = pageBounds[i + 1] - pageBounds[i];
    p2p->encodingsPageOffset =
        p2p->entryPageOffset + p2p->entryCount * sizeof(uint32_t);
    p2p->encodingsCount = 0;
    auto *ep = reinterpret_cast<uint32_t *>(&p2p[1]);
    auto cuPtrVectorIt = pageBounds[i];
    uintptr_t functionAddressBase = (*cuPtrVectorIt)->functionAddress;
    while (cuPtrVectorIt < pageBounds[i + 1]) {
      const CompactUnwindEntry64 *cuep = *cuPtrVectorIt++;
      size_t cueIndex = commonEncodingIndexes.lookup(cuep->encoding);
      *ep++ = ((cueIndex << UNWIND_INFO_COMPRESSED_ENTRY_FUNC_OFFSET_BITS) |
               (cuep->functionAddress - functionAddressBase));
    }
    p2p =
        reinterpret_cast<unwind_info_compressed_second_level_page_header *>(ep);
  }
  assert(getSize() ==
         static_cast<size_t>((reinterpret_cast<uint8_t *>(p2p) - buf)));
}
