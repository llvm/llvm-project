//===- UnwindInfoSection.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnwindInfoSection.h"
#include "ConcatOutputSection.h"
#include "Config.h"
#include "InputSection.h"
#include "OutputSection.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"

#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/STLExtras.h"
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

// TODO(gkm): prune __eh_frame entries superseded by __unwind_info, PR50410
// TODO(gkm): how do we align the 2nd-level pages?

using EncodingMap = DenseMap<compact_unwind_encoding_t, size_t>;

struct SecondLevelPage {
  uint32_t kind;
  size_t entryIndex;
  size_t entryCount;
  size_t byteCount;
  std::vector<compact_unwind_encoding_t> localEncodings;
  EncodingMap localEncodingIndexes;
};

template <class Ptr>
class UnwindInfoSectionImpl final : public UnwindInfoSection {
public:
  void prepareRelocations(ConcatInputSection *) override;
  void addInput(ConcatInputSection *) override;
  void finalize() override;
  void writeTo(uint8_t *buf) const override;

private:
  std::vector<std::pair<compact_unwind_encoding_t, size_t>> commonEncodings;
  EncodingMap commonEncodingIndexes;
  // Indices of personality functions within the GOT.
  std::vector<uint32_t> personalities;
  SmallDenseMap<std::pair<InputSection *, uint64_t /* addend */>, Symbol *>
      personalityTable;
  std::vector<unwind_info_section_header_lsda_index_entry> lsdaEntries;
  // Map of function offset (from the image base) to an index within the LSDA
  // array.
  DenseMap<uint32_t, uint32_t> functionToLsdaIndex;
  std::vector<CompactUnwindEntry<Ptr>> cuVector;
  std::vector<CompactUnwindEntry<Ptr> *> cuPtrVector;
  std::vector<SecondLevelPage> secondLevelPages;
  uint64_t level2PagesOffset = 0;
};

UnwindInfoSection::UnwindInfoSection()
    : SyntheticSection(segment_names::text, section_names::unwindInfo) {
  align = 4;
  compactUnwindSection =
      make<ConcatOutputSection>(section_names::compactUnwind);
}

void UnwindInfoSection::prepareRelocations() {
  for (ConcatInputSection *isec : compactUnwindSection->inputs)
    prepareRelocations(isec);
}

template <class Ptr>
void UnwindInfoSectionImpl<Ptr>::addInput(ConcatInputSection *isec) {
  assert(isec->getSegName() == segment_names::ld &&
         isec->getName() == section_names::compactUnwind);
  isec->parent = compactUnwindSection;
  compactUnwindSection->addInput(isec);
}

// Compact unwind relocations have different semantics, so we handle them in a
// separate code path from regular relocations. First, we do not wish to add
// rebase opcodes for __LD,__compact_unwind, because that section doesn't
// actually end up in the final binary. Second, personality pointers always
// reside in the GOT and must be treated specially.
template <class Ptr>
void UnwindInfoSectionImpl<Ptr>::prepareRelocations(ConcatInputSection *isec) {
  assert(!isec->shouldOmitFromOutput() &&
         "__compact_unwind section should not be omitted");

  // FIXME: Make this skip relocations for CompactUnwindEntries that
  // point to dead-stripped functions. That might save some amount of
  // work. But since there are usually just few personality functions
  // that are referenced from many places, at least some of them likely
  // live, it wouldn't reduce number of got entries.
  for (size_t i = 0; i < isec->relocs.size(); ++i) {
    Reloc &r = isec->relocs[i];
    assert(target->hasAttr(r.type, RelocAttrBits::UNSIGNED));

    if (r.offset % sizeof(CompactUnwindEntry<Ptr>) == 0) {
      InputSection *referentIsec;
      if (auto *isec = r.referent.dyn_cast<InputSection *>())
        referentIsec = isec;
      else
        referentIsec = cast<Defined>(r.referent.dyn_cast<Symbol *>())->isec;

      if (!cast<ConcatInputSection>(referentIsec)->shouldOmitFromOutput())
        allEntriesAreOmitted = false;
      continue;
    }

    if (r.offset % sizeof(CompactUnwindEntry<Ptr>) !=
        offsetof(CompactUnwindEntry<Ptr>, personality))
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
      assert(!isCoalescedWeak(referentIsec));
      // Personality functions can be referenced via section relocations
      // if they live in the same object file. Create placeholder synthetic
      // symbols for them in the GOT.
      Symbol *&s = personalityTable[{referentIsec, r.addend}];
      if (s == nullptr) {
        // This runs after dead stripping, so the noDeadStrip argument does not
        // matter.
        s = make<Defined>("<internal>", /*file=*/nullptr, referentIsec,
                          r.addend, /*size=*/0, /*isWeakDef=*/false,
                          /*isExternal=*/false, /*isPrivateExtern=*/false,
                          /*isThumb=*/false, /*isReferencedDynamically=*/false,
                          /*noDeadStrip=*/false);
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
static ConcatInputSection *checkTextSegment(InputSection *isec) {
  if (isec->getSegName() != segment_names::text)
    error("compact unwind references address in " + toString(isec) +
          " which is not in segment __TEXT");
  // __text should always be a ConcatInputSection.
  return cast<ConcatInputSection>(isec);
}

template <class Ptr>
constexpr Ptr TombstoneValue = std::numeric_limits<Ptr>::max();

// We need to apply the relocations to the pre-link compact unwind section
// before converting it to post-link form. There should only be absolute
// relocations here: since we are not emitting the pre-link CU section, there
// is no source address to make a relative location meaningful.
template <class Ptr>
static void
relocateCompactUnwind(ConcatOutputSection *compactUnwindSection,
                      std::vector<CompactUnwindEntry<Ptr>> &cuVector) {
  for (const ConcatInputSection *isec : compactUnwindSection->inputs) {
    assert(isec->parent == compactUnwindSection);

    uint8_t *buf =
        reinterpret_cast<uint8_t *>(cuVector.data()) + isec->outSecOff;
    memcpy(buf, isec->data.data(), isec->data.size());

    for (const Reloc &r : isec->relocs) {
      uint64_t referentVA = TombstoneValue<Ptr>;
      if (auto *referentSym = r.referent.dyn_cast<Symbol *>()) {
        if (!isa<Undefined>(referentSym)) {
          if (auto *defined = dyn_cast<Defined>(referentSym))
            checkTextSegment(defined->isec);
          // At this point in the link, we may not yet know the final address of
          // the GOT, so we just encode the index. We make it a 1-based index so
          // that we can distinguish the null pointer case.
          referentVA = referentSym->gotIndex + 1;
        }
      } else {
        auto *referentIsec = r.referent.get<InputSection *>();
        ConcatInputSection *concatIsec = checkTextSegment(referentIsec);
        if (!concatIsec->shouldOmitFromOutput())
          referentVA = referentIsec->getVA(r.addend);
      }
      writeAddress(buf + r.offset, referentVA, r.length);
    }
  }
}

// There should only be a handful of unique personality pointers, so we can
// encode them as 2-bit indices into a small array.
template <class Ptr>
static void
encodePersonalities(const std::vector<CompactUnwindEntry<Ptr> *> &cuPtrVector,
                    std::vector<uint32_t> &personalities) {
  for (CompactUnwindEntry<Ptr> *cu : cuPtrVector) {
    if (cu->personality == 0)
      continue;
    // Linear search is fast enough for a small array.
    auto it = find(personalities, cu->personality);
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

// __unwind_info stores unwind data for address ranges. If several
// adjacent functions have the same unwind encoding, LSDA, and personality
// function, they share one unwind entry. For this to work, functions without
// unwind info need explicit "no unwind info" unwind entries -- else the
// unwinder would think they have the unwind info of the closest function
// with unwind info right before in the image.
template <class Ptr>
static void addEntriesForFunctionsWithoutUnwindInfo(
    std::vector<CompactUnwindEntry<Ptr>> &cuVector) {
  DenseSet<Ptr> hasUnwindInfo;
  for (CompactUnwindEntry<Ptr> &cuEntry : cuVector)
    if (cuEntry.functionAddress != TombstoneValue<Ptr>)
      hasUnwindInfo.insert(cuEntry.functionAddress);

  // Add explicit "has no unwind info" entries for all global and local symbols
  // without unwind info.
  auto markNoUnwindInfo = [&cuVector, &hasUnwindInfo](const Defined *d) {
    if (d->isLive() && d->isec && isCodeSection(d->isec)) {
      Ptr ptr = d->getVA();
      if (!hasUnwindInfo.count(ptr))
        cuVector.push_back({ptr, 0, 0, 0, 0});
    }
  };
  for (Symbol *sym : symtab->getSymbols())
    if (auto *d = dyn_cast<Defined>(sym))
      markNoUnwindInfo(d);
  for (const InputFile *file : inputFiles)
    if (auto *objFile = dyn_cast<ObjFile>(file))
      for (Symbol *sym : objFile->symbols)
        if (auto *d = dyn_cast_or_null<Defined>(sym))
          if (!d->isExternal())
            markNoUnwindInfo(d);
}

static bool canFoldEncoding(compact_unwind_encoding_t encoding) {
  // From compact_unwind_encoding.h:
  //  UNWIND_X86_64_MODE_STACK_IND:
  //  A "frameless" (RBP not used as frame pointer) function large constant
  //  stack size.  This case is like the previous, except the stack size is too
  //  large to encode in the compact unwind encoding.  Instead it requires that
  //  the function contains "subq $nnnnnnnn,RSP" in its prolog.  The compact
  //  encoding contains the offset to the nnnnnnnn value in the function in
  //  UNWIND_X86_64_FRAMELESS_STACK_SIZE.
  // Since this means the unwinder has to look at the `subq` in the function
  // of the unwind info's unwind address, two functions that have identical
  // unwind info can't be folded if it's using this encoding since both
  // entries need unique addresses.
  static_assert(UNWIND_X86_64_MODE_MASK == UNWIND_X86_MODE_MASK, "");
  static_assert(UNWIND_X86_64_MODE_STACK_IND == UNWIND_X86_MODE_STACK_IND, "");
  if ((target->cpuType == CPU_TYPE_X86_64 || target->cpuType == CPU_TYPE_X86) &&
      (encoding & UNWIND_X86_64_MODE_MASK) == UNWIND_X86_64_MODE_STACK_IND) {
    // FIXME: Consider passing in the two function addresses and getting
    // their two stack sizes off the `subq` and only returning false if they're
    // actually different.
    return false;
  }
  return true;
}

// Scan the __LD,__compact_unwind entries and compute the space needs of
// __TEXT,__unwind_info and __TEXT,__eh_frame
template <class Ptr> void UnwindInfoSectionImpl<Ptr>::finalize() {
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
  assert(compactUnwindSection->getSize() % sizeof(CompactUnwindEntry<Ptr>) ==
         0);
  size_t cuCount =
      compactUnwindSection->getSize() / sizeof(CompactUnwindEntry<Ptr>);
  cuVector.resize(cuCount);
  relocateCompactUnwind(compactUnwindSection, cuVector);

  addEntriesForFunctionsWithoutUnwindInfo(cuVector);

  // Rather than sort & fold the 32-byte entries directly, we create a
  // vector of pointers to entries and sort & fold that instead.
  cuPtrVector.reserve(cuVector.size());
  for (CompactUnwindEntry<Ptr> &cuEntry : cuVector)
    cuPtrVector.emplace_back(&cuEntry);
  llvm::sort(cuPtrVector, [](const CompactUnwindEntry<Ptr> *a,
                             const CompactUnwindEntry<Ptr> *b) {
    return a->functionAddress < b->functionAddress;
  });

  // Dead-stripped functions get a functionAddress of TombstoneValue in
  // relocateCompactUnwind(). Filter them out here.
  // FIXME: This doesn't yet collect associated data like LSDAs kept
  // alive only by a now-removed CompactUnwindEntry or other comdat-like
  // data (`kindNoneGroupSubordinate*` in ld64).
  CompactUnwindEntry<Ptr> tombstone;
  tombstone.functionAddress = TombstoneValue<Ptr>;
  cuPtrVector.erase(
      std::lower_bound(cuPtrVector.begin(), cuPtrVector.end(), &tombstone,
                       [](const CompactUnwindEntry<Ptr> *a,
                          const CompactUnwindEntry<Ptr> *b) {
                         return a->functionAddress < b->functionAddress;
                       }),
      cuPtrVector.end());

  // If there are no entries left after adding explicit "no unwind info"
  // entries and removing entries for dead-stripped functions, don't write
  // an __unwind_info section at all.
  assert(allEntriesAreOmitted == cuPtrVector.empty());
  if (cuPtrVector.empty())
    return;

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
           (*foldBegin)->lsda == (*foldEnd)->lsda &&
           canFoldEncoding((*foldEnd)->encoding))
      ;
    *foldWrite++ = *foldBegin;
    foldBegin = foldEnd;
  }
  cuPtrVector.erase(foldWrite, cuPtrVector.end());

  encodePersonalities(cuPtrVector, personalities);

  // Count frequencies of the folded encodings
  EncodingMap encodingFrequencies;
  for (const CompactUnwindEntry<Ptr> *cuPtrEntry : cuPtrVector)
    encodingFrequencies[cuPtrEntry->encoding]++;

  // Make a vector of encodings, sorted by descending frequency
  for (const auto &frequency : encodingFrequencies)
    commonEncodings.emplace_back(frequency);
  llvm::sort(commonEncodings,
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
    SecondLevelPage &page = secondLevelPages.back();
    page.entryIndex = i;
    uintptr_t functionAddressMax =
        cuPtrVector[i]->functionAddress + COMPRESSED_ENTRY_FUNC_OFFSET_MASK;
    size_t n = commonEncodings.size();
    size_t wordsRemaining =
        SECOND_LEVEL_PAGE_WORDS -
        sizeof(unwind_info_compressed_second_level_page_header) /
            sizeof(uint32_t);
    while (wordsRemaining >= 1 && i < cuPtrVector.size()) {
      const CompactUnwindEntry<Ptr> *cuPtr = cuPtrVector[i];
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

  for (const CompactUnwindEntry<Ptr> *cu : cuPtrVector) {
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

template <class Ptr>
void UnwindInfoSectionImpl<Ptr>::writeTo(uint8_t *buf) const {
  assert(!cuPtrVector.empty() && "call only if there is unwind info");

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
    *i32p++ =
        in.got->addr + (personality - 1) * target->wordSize - in.header->addr;

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
  const CompactUnwindEntry<Ptr> &cuEnd = *cuPtrVector.back();
  assert(cuEnd.functionAddress != TombstoneValue<Ptr>);
  iep->functionOffset =
      cuEnd.functionAddress - in.header->addr + cuEnd.functionLength;
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
        const CompactUnwindEntry<Ptr> *cuep = cuPtrVector[page.entryIndex + i];
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
        const CompactUnwindEntry<Ptr> *cuep = cuPtrVector[page.entryIndex + i];
        *ep++ = cuep->functionAddress;
        *ep++ = cuep->encoding;
      }
    }
    pp += SECOND_LEVEL_PAGE_WORDS;
  }
}

UnwindInfoSection *macho::makeUnwindInfoSection() {
  if (target->wordSize == 8)
    return make<UnwindInfoSectionImpl<uint64_t>>();
  else
    return make<UnwindInfoSectionImpl<uint32_t>>();
}
