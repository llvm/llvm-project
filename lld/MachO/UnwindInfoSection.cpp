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
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"

#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/MachO.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace lld;
using namespace lld::macho;

#define COMMON_ENCODINGS_MAX 127
#define COMPACT_ENCODINGS_MAX 256

#define SECOND_LEVEL_PAGE_BYTES 4096
#define SECOND_LEVEL_PAGE_WORDS (SECOND_LEVEL_PAGE_BYTES / sizeof(uint32_t))
#define REGULAR_SECOND_LEVEL_ENTRIES_MAX                                       \
  ((SECOND_LEVEL_PAGE_BYTES -                                                  \
    sizeof(unwind_info_regular_second_level_page_header)) /                    \
   sizeof(unwind_info_regular_second_level_entry))
#define COMPRESSED_SECOND_LEVEL_ENTRIES_MAX                                    \
  ((SECOND_LEVEL_PAGE_BYTES -                                                  \
    sizeof(unwind_info_compressed_second_level_page_header)) /                 \
   sizeof(uint32_t))

#define COMPRESSED_ENTRY_FUNC_OFFSET_BITS 24
#define COMPRESSED_ENTRY_FUNC_OFFSET_MASK                                      \
  UNWIND_INFO_COMPRESSED_ENTRY_FUNC_OFFSET(~0)

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
//
// Refer to the definition of unwind_info_section_header in
// compact_unwind_encoding.h for an overview of the format we are encoding
// here.

// TODO(gkm): prune __eh_frame entries superseded by __unwind_info
// TODO(gkm): how do we align the 2nd-level pages?

UnwindInfoSection::UnwindInfoSection()
    : SyntheticSection(segment_names::text, section_names::unwindInfo) {
  align = 4; // mimic ld64
}

bool UnwindInfoSection::isNeeded() const {
  return (compactUnwindSection != nullptr);
}

SmallDenseMap<std::pair<InputSection *, uint64_t /* addend */>, Symbol *>
    personalityTable;

// Compact unwind relocations have different semantics, so we handle them in a
// separate code path from regular relocations. First, we do not wish to add
// rebase opcodes for __LD,__compact_unwind, because that section doesn't
// actually end up in the final binary. Second, personality pointers always
// reside in the GOT and must be treated specially.
void macho::prepareCompactUnwind(InputSection *isec) {
  assert(isec->segname == segment_names::ld &&
         isec->name == section_names::compactUnwind);

  for (Reloc &r : isec->relocs) {
    assert(target->hasAttr(r.type, RelocAttrBits::UNSIGNED));
    if (r.offset % sizeof(CompactUnwindEntry64) !=
        offsetof(struct CompactUnwindEntry64, personality))
      continue;

    if (auto *s = r.referent.dyn_cast<Symbol *>()) {
      if (auto *undefined = dyn_cast<Undefined>(s)) {
        treatUndefinedSymbol(*undefined);
        // treatUndefinedSymbol() can replace s with a DylibSymbol; re-check.
        if (isa<Undefined>(s))
          continue;
      }
      if (auto *defined = dyn_cast<Defined>(s)) {
        // Check if we have created a synthetic symbol at the same address.
        Symbol *&personality =
            personalityTable[{defined->isec, defined->value}];
        if (personality == nullptr) {
          personality = defined;
          in.got->addEntry(defined);
        } else if (personality != defined) {
          r.referent = personality;
        }
        continue;
      }
      assert(isa<DylibSymbol>(s));
      in.got->addEntry(s);
      continue;
    }

    if (auto *referentIsec = r.referent.dyn_cast<InputSection *>()) {
      // Personality functions can be referenced via section relocations
      // if they live in the same object file. Create placeholder synthetic
      // symbols for them in the GOT.
      Symbol *&s = personalityTable[{referentIsec, r.addend}];
      if (s == nullptr) {
        s = make<Defined>("<internal>", nullptr, referentIsec, r.addend, 0,
                          false, false, false);
        in.got->addEntry(s);
      }
      r.referent = s;
      r.addend = 0;
    }
  }
}

// Unwind info lives in __DATA, and finalization of __TEXT will occur before
// finalization of __DATA. Moreover, the finalization of unwind info depends on
// the exact addresses that it references. So it is safe for compact unwind to
// reference addresses in __TEXT, but not addresses in any other segment.
static void checkTextSegment(InputSection *isec) {
  if (isec->segname != segment_names::text)
    error("compact unwind references address in " + toString(isec) +
          " which is not in segment __TEXT");
}

// We need to apply the relocations to the pre-link compact unwind section
// before converting it to post-link form. There should only be absolute
// relocations here: since we are not emitting the pre-link CU section, there
// is no source address to make a relative location meaningful.
static void relocateCompactUnwind(MergedOutputSection *compactUnwindSection,
                                  std::vector<CompactUnwindEntry64> &cuVector) {
  for (const InputSection *isec : compactUnwindSection->inputs) {
    uint8_t *buf =
        reinterpret_cast<uint8_t *>(cuVector.data()) + isec->outSecFileOff;
    memcpy(buf, isec->data.data(), isec->data.size());

    for (const Reloc &r : isec->relocs) {
      uint64_t referentVA = 0;
      if (auto *referentSym = r.referent.dyn_cast<Symbol *>()) {
        if (!isa<Undefined>(referentSym)) {
          assert(referentSym->isInGot());
          if (auto *defined = dyn_cast<Defined>(referentSym))
            checkTextSegment(defined->isec);
          // At this point in the link, we may not yet know the final address of
          // the GOT, so we just encode the index. We make it a 1-based index so
          // that we can distinguish the null pointer case.
          referentVA = referentSym->gotIndex + 1;
        }
      } else if (auto *referentIsec = r.referent.dyn_cast<InputSection *>()) {
        checkTextSegment(referentIsec);
        referentVA = referentIsec->getVA() + r.addend;
      }
      support::endian::write64le(buf + r.offset, referentVA);
    }
  }
}

// There should only be a handful of unique personality pointers, so we can
// encode them as 2-bit indices into a small array.
void encodePersonalities(const std::vector<CompactUnwindEntry64 *> &cuPtrVector,
                         std::vector<uint32_t> &personalities) {
  for (CompactUnwindEntry64 *cu : cuPtrVector) {
    if (cu->personality == 0)
      continue;
    uint32_t personalityOffset = cu->personality - in.header->addr;
    // Linear search is fast enough for a small array.
    auto it = find(personalities, personalityOffset);
    uint32_t personalityIndex; // 1-based index
    if (it != personalities.end()) {
      personalityIndex = std::distance(personalities.begin(), it) + 1;
    } else {
      personalities.push_back(cu->personality);
      personalityIndex = personalities.size();
    }
    cu->encoding |=
        personalityIndex << countTrailingZeros(
            static_cast<compact_unwind_encoding_t>(UNWIND_PERSONALITY_MASK));
  }
  if (personalities.size() > 3)
    error("too many personalities (" + std::to_string(personalities.size()) +
          ") for compact unwind to encode");
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
  relocateCompactUnwind(compactUnwindSection, cuVector);

  // Rather than sort & fold the 32-byte entries directly, we create a
  // vector of pointers to entries and sort & fold that instead.
  cuPtrVector.reserve(cuCount);
  for (CompactUnwindEntry64 &cuEntry : cuVector)
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

  encodePersonalities(cuPtrVector, personalities);

  // Count frequencies of the folded encodings
  EncodingMap encodingFrequencies;
  for (const CompactUnwindEntry64 *cuPtrEntry : cuPtrVector)
    encodingFrequencies[cuPtrEntry->encoding]++;

  // Make a vector of encodings, sorted by descending frequency
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

  // Truncate the vector to 127 elements.
  // Common encoding indexes are limited to 0..126, while encoding
  // indexes 127..255 are local to each second-level page
  if (commonEncodings.size() > COMMON_ENCODINGS_MAX)
    commonEncodings.resize(COMMON_ENCODINGS_MAX);

  // Create a map from encoding to common-encoding-table index
  for (size_t i = 0; i < commonEncodings.size(); i++)
    commonEncodingIndexes[commonEncodings[i].first] = i;

  // Split folded encodings into pages, where each page is limited by ...
  // (a) 4 KiB capacity
  // (b) 24-bit difference between first & final function address
  // (c) 8-bit compact-encoding-table index,
  //     for which 0..126 references the global common-encodings table,
  //     and 127..255 references a local per-second-level-page table.
  // First we try the compact format and determine how many entries fit.
  // If more entries fit in the regular format, we use that.
  for (size_t i = 0; i < cuPtrVector.size();) {
    secondLevelPages.emplace_back();
    UnwindInfoSection::SecondLevelPage &page = secondLevelPages.back();
    page.entryIndex = i;
    uintptr_t functionAddressMax =
        cuPtrVector[i]->functionAddress + COMPRESSED_ENTRY_FUNC_OFFSET_MASK;
    size_t n = commonEncodings.size();
    size_t wordsRemaining =
        SECOND_LEVEL_PAGE_WORDS -
        sizeof(unwind_info_compressed_second_level_page_header) /
            sizeof(uint32_t);
    while (wordsRemaining >= 1 && i < cuPtrVector.size()) {
      const CompactUnwindEntry64 *cuPtr = cuPtrVector[i];
      if (cuPtr->functionAddress >= functionAddressMax) {
        break;
      } else if (commonEncodingIndexes.count(cuPtr->encoding) ||
                 page.localEncodingIndexes.count(cuPtr->encoding)) {
        i++;
        wordsRemaining--;
      } else if (wordsRemaining >= 2 && n < COMPACT_ENCODINGS_MAX) {
        page.localEncodings.emplace_back(cuPtr->encoding);
        page.localEncodingIndexes[cuPtr->encoding] = n++;
        i++;
        wordsRemaining -= 2;
      } else {
        break;
      }
    }
    page.entryCount = i - page.entryIndex;

    // If this is not the final page, see if it's possible to fit more
    // entries by using the regular format. This can happen when there
    // are many unique encodings, and we we saturated the local
    // encoding table early.
    if (i < cuPtrVector.size() &&
        page.entryCount < REGULAR_SECOND_LEVEL_ENTRIES_MAX) {
      page.kind = UNWIND_SECOND_LEVEL_REGULAR;
      page.entryCount = std::min(REGULAR_SECOND_LEVEL_ENTRIES_MAX,
                                 cuPtrVector.size() - page.entryIndex);
      i = page.entryIndex + page.entryCount;
    } else {
      page.kind = UNWIND_SECOND_LEVEL_COMPRESSED;
    }
  }

  for (const CompactUnwindEntry64 *cu : cuPtrVector) {
    uint32_t functionOffset = cu->functionAddress - in.header->addr;
    functionToLsdaIndex[functionOffset] = lsdaEntries.size();
    if (cu->lsda != 0)
      lsdaEntries.push_back(
          {functionOffset, static_cast<uint32_t>(cu->lsda - in.header->addr)});
  }

  // compute size of __TEXT,__unwind_info section
  level2PagesOffset =
      sizeof(unwind_info_section_header) +
      commonEncodings.size() * sizeof(uint32_t) +
      personalities.size() * sizeof(uint32_t) +
      // The extra second-level-page entry is for the sentinel
      (secondLevelPages.size() + 1) *
          sizeof(unwind_info_section_header_index_entry) +
      lsdaEntries.size() * sizeof(unwind_info_section_header_lsda_index_entry);
  unwindInfoSize =
      level2PagesOffset + secondLevelPages.size() * SECOND_LEVEL_PAGE_BYTES;
}

// All inputs are relocated and output addresses are known, so write!

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
  uip->indexCount = secondLevelPages.size() + 1;

  // Common encodings
  auto *i32p = reinterpret_cast<uint32_t *>(&uip[1]);
  for (const auto &encoding : commonEncodings)
    *i32p++ = encoding.first;

  // Personalities
  for (const uint32_t &personality : personalities)
    *i32p++ = in.got->addr + (personality - 1) * target->wordSize;

  // Level-1 index
  uint32_t lsdaOffset =
      uip->indexSectionOffset +
      uip->indexCount * sizeof(unwind_info_section_header_index_entry);
  uint64_t l2PagesOffset = level2PagesOffset;
  auto *iep = reinterpret_cast<unwind_info_section_header_index_entry *>(i32p);
  for (const SecondLevelPage &page : secondLevelPages) {
    iep->functionOffset =
        cuPtrVector[page.entryIndex]->functionAddress - in.header->addr;
    iep->secondLevelPagesSectionOffset = l2PagesOffset;
    iep->lsdaIndexArraySectionOffset =
        lsdaOffset + functionToLsdaIndex.lookup(iep->functionOffset) *
                         sizeof(unwind_info_section_header_lsda_index_entry);
    iep++;
    l2PagesOffset += SECOND_LEVEL_PAGE_BYTES;
  }
  // Level-1 sentinel
  const CompactUnwindEntry64 &cuEnd = cuVector.back();
  iep->functionOffset = cuEnd.functionAddress + cuEnd.functionLength;
  iep->secondLevelPagesSectionOffset = 0;
  iep->lsdaIndexArraySectionOffset =
      lsdaOffset +
      lsdaEntries.size() * sizeof(unwind_info_section_header_lsda_index_entry);
  iep++;

  // LSDAs
  size_t lsdaBytes =
      lsdaEntries.size() * sizeof(unwind_info_section_header_lsda_index_entry);
  if (lsdaBytes > 0)
    memcpy(iep, lsdaEntries.data(), lsdaBytes);

  // Level-2 pages
  auto *pp = reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(iep) +
                                          lsdaBytes);
  for (const SecondLevelPage &page : secondLevelPages) {
    if (page.kind == UNWIND_SECOND_LEVEL_COMPRESSED) {
      uintptr_t functionAddressBase =
          cuPtrVector[page.entryIndex]->functionAddress;
      auto *p2p =
          reinterpret_cast<unwind_info_compressed_second_level_page_header *>(
              pp);
      p2p->kind = page.kind;
      p2p->entryPageOffset =
          sizeof(unwind_info_compressed_second_level_page_header);
      p2p->entryCount = page.entryCount;
      p2p->encodingsPageOffset =
          p2p->entryPageOffset + p2p->entryCount * sizeof(uint32_t);
      p2p->encodingsCount = page.localEncodings.size();
      auto *ep = reinterpret_cast<uint32_t *>(&p2p[1]);
      for (size_t i = 0; i < page.entryCount; i++) {
        const CompactUnwindEntry64 *cuep = cuPtrVector[page.entryIndex + i];
        auto it = commonEncodingIndexes.find(cuep->encoding);
        if (it == commonEncodingIndexes.end())
          it = page.localEncodingIndexes.find(cuep->encoding);
        *ep++ = (it->second << COMPRESSED_ENTRY_FUNC_OFFSET_BITS) |
                (cuep->functionAddress - functionAddressBase);
      }
      if (page.localEncodings.size() != 0)
        memcpy(ep, page.localEncodings.data(),
               page.localEncodings.size() * sizeof(uint32_t));
    } else {
      auto *p2p =
          reinterpret_cast<unwind_info_regular_second_level_page_header *>(pp);
      p2p->kind = page.kind;
      p2p->entryPageOffset =
          sizeof(unwind_info_regular_second_level_page_header);
      p2p->entryCount = page.entryCount;
      auto *ep = reinterpret_cast<uint32_t *>(&p2p[1]);
      for (size_t i = 0; i < page.entryCount; i++) {
        const CompactUnwindEntry64 *cuep = cuPtrVector[page.entryIndex + i];
        *ep++ = cuep->functionAddress;
        *ep++ = cuep->encoding;
      }
    }
    pp += SECOND_LEVEL_PAGE_WORDS;
  }
}
