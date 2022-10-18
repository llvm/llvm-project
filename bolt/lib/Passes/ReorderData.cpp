//===- bolt/Passes/ReorderSection.cpp - Reordering of section data --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements ReorderData class.
//
//===----------------------------------------------------------------------===//

// TODO:
// - make sure writeable data isn't put on same cache line unless temporally
// local
// - estimate temporal locality by looking at CFG?

#include "bolt/Passes/ReorderData.h"
#include <algorithm>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "reorder-data"

using namespace llvm;
using namespace bolt;

namespace opts {
extern cl::OptionCategory BoltCategory;
extern cl::OptionCategory BoltOptCategory;
extern cl::opt<JumpTableSupportLevel> JumpTables;

static cl::opt<bool>
    PrintReorderedData("print-reordered-data",
                       cl::desc("print section contents after reordering"),
                       cl::Hidden, cl::cat(BoltCategory));

cl::list<std::string>
ReorderData("reorder-data",
  cl::CommaSeparated,
  cl::desc("list of sections to reorder"),
  cl::value_desc("section1,section2,section3,..."),
  cl::cat(BoltOptCategory));

enum ReorderAlgo : char {
  REORDER_COUNT         = 0,
  REORDER_FUNCS         = 1
};

static cl::opt<ReorderAlgo>
ReorderAlgorithm("reorder-data-algo",
  cl::desc("algorithm used to reorder data sections"),
  cl::init(REORDER_COUNT),
  cl::values(
    clEnumValN(REORDER_COUNT,
      "count",
      "sort hot data by read counts"),
    clEnumValN(REORDER_FUNCS,
      "funcs",
      "sort hot data by hot function usage and count")),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<unsigned>
    ReorderDataMaxSymbols("reorder-data-max-symbols",
                          cl::desc("maximum number of symbols to reorder"),
                          cl::init(std::numeric_limits<unsigned>::max()),
                          cl::cat(BoltOptCategory));

static cl::opt<unsigned> ReorderDataMaxBytes(
    "reorder-data-max-bytes", cl::desc("maximum number of bytes to reorder"),
    cl::init(std::numeric_limits<unsigned>::max()), cl::cat(BoltOptCategory));

static cl::list<std::string>
ReorderSymbols("reorder-symbols",
  cl::CommaSeparated,
  cl::desc("list of symbol names that can be reordered"),
  cl::value_desc("symbol1,symbol2,symbol3,..."),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::list<std::string>
SkipSymbols("reorder-skip-symbols",
  cl::CommaSeparated,
  cl::desc("list of symbol names that cannot be reordered"),
  cl::value_desc("symbol1,symbol2,symbol3,..."),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<bool> ReorderInplace("reorder-data-inplace",
                                    cl::desc("reorder data sections in place"),

                                    cl::cat(BoltOptCategory));
}

namespace llvm {
namespace bolt {

namespace {

static constexpr uint16_t MinAlignment = 16;

bool isSupported(const BinarySection &BS) { return BS.isData() && !BS.isTLS(); }

bool filterSymbol(const BinaryData *BD) {
  if (!BD->isAtomic() || BD->isJumpTable() || !BD->isMoveable())
    return false;

  bool IsValid = true;

  if (!opts::ReorderSymbols.empty()) {
    IsValid = llvm::any_of(opts::ReorderSymbols, [&](const std::string &Name) {
      return BD->hasName(Name);
    });
  }

  if (!IsValid)
    return false;

  if (!opts::SkipSymbols.empty()) {
    for (const std::string &Name : opts::SkipSymbols) {
      if (BD->hasName(Name)) {
        IsValid = false;
        break;
      }
    }
  }

  return IsValid;
}

} // namespace

using DataOrder = ReorderData::DataOrder;

void ReorderData::printOrder(const BinarySection &Section,
                             DataOrder::const_iterator Begin,
                             DataOrder::const_iterator End) const {
  uint64_t TotalSize = 0;
  bool PrintHeader = false;
  while (Begin != End) {
    const BinaryData *BD = Begin->first;

    if (!PrintHeader) {
      outs() << "BOLT-INFO: Hot global symbols for " << Section.getName()
             << ":\n";
      PrintHeader = true;
    }

    outs() << "BOLT-INFO: " << *BD << ", moveable=" << BD->isMoveable()
           << format(", weight=%.5f\n", double(Begin->second) / BD->getSize());

    TotalSize += BD->getSize();
    ++Begin;
  }
  if (TotalSize)
    outs() << "BOLT-INFO: Total hot symbol size = " << TotalSize << "\n";
}

DataOrder ReorderData::baseOrder(BinaryContext &BC,
                                 const BinarySection &Section) const {
  DataOrder Order;
  for (auto &Entry : BC.getBinaryDataForSection(Section)) {
    BinaryData *BD = Entry.second;
    if (!BD->isAtomic()) // skip sub-symbols
      continue;
    auto BDCI = BinaryDataCounts.find(BD);
    uint64_t BDCount = BDCI == BinaryDataCounts.end() ? 0 : BDCI->second;
    Order.emplace_back(BD, BDCount);
  }
  return Order;
}

void ReorderData::assignMemData(BinaryContext &BC) {
  // Map of sections (or heap/stack) to count/size.
  StringMap<uint64_t> Counts;
  StringMap<uint64_t> JumpTableCounts;
  uint64_t TotalCount = 0;
  for (auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction &BF = BFI.second;
    if (!BF.hasMemoryProfile())
      continue;

    for (const BinaryBasicBlock &BB : BF) {
      for (const MCInst &Inst : BB) {
        auto ErrorOrMemAccessProfile =
            BC.MIB->tryGetAnnotationAs<MemoryAccessProfile>(
                Inst, "MemoryAccessProfile");
        if (!ErrorOrMemAccessProfile)
          continue;

        const MemoryAccessProfile &MemAccessProfile =
            ErrorOrMemAccessProfile.get();
        for (const AddressAccess &AccessInfo :
             MemAccessProfile.AddressAccessInfo) {
          if (BinaryData *BD = AccessInfo.MemoryObject) {
            BinaryDataCounts[BD->getAtomicRoot()] += AccessInfo.Count;
            Counts[BD->getSectionName()] += AccessInfo.Count;
            if (BD->getAtomicRoot()->isJumpTable())
              JumpTableCounts[BD->getSectionName()] += AccessInfo.Count;
          } else {
            Counts["Heap/stack"] += AccessInfo.Count;
          }
          TotalCount += AccessInfo.Count;
        }
      }
    }
  }

  if (!Counts.empty()) {
    outs() << "BOLT-INFO: Memory stats breakdown:\n";
    for (StringMapEntry<uint64_t> &Entry : Counts) {
      StringRef Section = Entry.first();
      const uint64_t Count = Entry.second;
      outs() << "BOLT-INFO:   " << Section << " = " << Count
             << format(" (%.1f%%)\n", 100.0 * Count / TotalCount);
      if (JumpTableCounts.count(Section) != 0) {
        const uint64_t JTCount = JumpTableCounts[Section];
        outs() << "BOLT-INFO:     jump tables = " << JTCount
               << format(" (%.1f%%)\n", 100.0 * JTCount / Count);
      }
    }
    outs() << "BOLT-INFO: Total memory events: " << TotalCount << "\n";
  }
}

/// Only consider moving data that is used by the hottest functions with
/// valid profiles.
std::pair<DataOrder, unsigned>
ReorderData::sortedByFunc(BinaryContext &BC, const BinarySection &Section,
                          std::map<uint64_t, BinaryFunction> &BFs) const {
  std::map<BinaryData *, std::set<BinaryFunction *>> BDtoFunc;
  std::map<BinaryData *, uint64_t> BDtoFuncCount;

  auto dataUses = [&BC](const BinaryFunction &BF, bool OnlyHot) {
    std::set<BinaryData *> Uses;
    for (const BinaryBasicBlock &BB : BF) {
      if (OnlyHot && BB.isCold())
        continue;

      for (const MCInst &Inst : BB) {
        auto ErrorOrMemAccessProfile =
            BC.MIB->tryGetAnnotationAs<MemoryAccessProfile>(
                Inst, "MemoryAccessProfile");
        if (!ErrorOrMemAccessProfile)
          continue;

        const MemoryAccessProfile &MemAccessProfile =
            ErrorOrMemAccessProfile.get();
        for (const AddressAccess &AccessInfo :
             MemAccessProfile.AddressAccessInfo) {
          if (AccessInfo.MemoryObject)
            Uses.insert(AccessInfo.MemoryObject);
        }
      }
    }
    return Uses;
  };

  for (auto &Entry : BFs) {
    BinaryFunction &BF = Entry.second;
    if (BF.hasValidProfile()) {
      for (BinaryData *BD : dataUses(BF, true)) {
        if (!BC.getFunctionForSymbol(BD->getSymbol())) {
          BDtoFunc[BD->getAtomicRoot()].insert(&BF);
          BDtoFuncCount[BD->getAtomicRoot()] += BF.getKnownExecutionCount();
        }
      }
    }
  }

  DataOrder Order = baseOrder(BC, Section);
  unsigned SplitPoint = Order.size();

  llvm::sort(
      Order,
      [&](const DataOrder::value_type &A, const DataOrder::value_type &B) {
        // Total execution counts of functions referencing BD.
        const uint64_t ACount = BDtoFuncCount[A.first];
        const uint64_t BCount = BDtoFuncCount[B.first];
        // Weight by number of loads/data size.
        const double AWeight = double(A.second) / A.first->getSize();
        const double BWeight = double(B.second) / B.first->getSize();
        return (ACount > BCount ||
                (ACount == BCount &&
                 (AWeight > BWeight ||
                  (AWeight == BWeight &&
                   A.first->getAddress() < B.first->getAddress()))));
      });

  for (unsigned Idx = 0; Idx < Order.size(); ++Idx) {
    if (!BDtoFuncCount[Order[Idx].first]) {
      SplitPoint = Idx;
      break;
    }
  }

  return std::make_pair(Order, SplitPoint);
}

std::pair<DataOrder, unsigned>
ReorderData::sortedByCount(BinaryContext &BC,
                           const BinarySection &Section) const {
  DataOrder Order = baseOrder(BC, Section);
  unsigned SplitPoint = Order.size();

  llvm::sort(Order, [](const DataOrder::value_type &A,
                       const DataOrder::value_type &B) {
    // Weight by number of loads/data size.
    const double AWeight = double(A.second) / A.first->getSize();
    const double BWeight = double(B.second) / B.first->getSize();
    return (AWeight > BWeight ||
            (AWeight == BWeight &&
             (A.first->getSize() < B.first->getSize() ||
              (A.first->getSize() == B.first->getSize() &&
               A.first->getAddress() < B.first->getAddress()))));
  });

  for (unsigned Idx = 0; Idx < Order.size(); ++Idx) {
    if (!Order[Idx].second) {
      SplitPoint = Idx;
      break;
    }
  }

  return std::make_pair(Order, SplitPoint);
}

// TODO
// add option for cache-line alignment (or just use cache-line when section
// is writeable)?
void ReorderData::setSectionOrder(BinaryContext &BC,
                                  BinarySection &OutputSection,
                                  DataOrder::iterator Begin,
                                  DataOrder::iterator End) {
  std::vector<BinaryData *> NewOrder;
  unsigned NumReordered = 0;
  uint64_t Offset = 0;
  uint64_t Count = 0;

  // Get the total count just for stats
  uint64_t TotalCount = 0;
  for (auto Itr = Begin; Itr != End; ++Itr)
    TotalCount += Itr->second;

  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: setSectionOrder for "
                    << OutputSection.getName() << "\n");

  for (; Begin != End; ++Begin) {
    BinaryData *BD = Begin->first;

    // We can't move certain symbols.
    if (!filterSymbol(BD))
      continue;

    ++NumReordered;
    if (NumReordered > opts::ReorderDataMaxSymbols) {
      if (!NewOrder.empty())
        LLVM_DEBUG(dbgs() << "BOLT-DEBUG: processing ending on symbol "
                          << *NewOrder.back() << "\n");
      break;
    }

    uint16_t Alignment = std::max(BD->getAlignment(), MinAlignment);
    Offset = alignTo(Offset, Alignment);

    if ((Offset + BD->getSize()) > opts::ReorderDataMaxBytes) {
      if (!NewOrder.empty())
        LLVM_DEBUG(dbgs() << "BOLT-DEBUG: processing ending on symbol "
                          << *NewOrder.back() << "\n");
      break;
    }

    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: " << BD->getName() << " @ 0x"
                      << Twine::utohexstr(Offset) << "\n");

    BD->setOutputLocation(OutputSection, Offset);

    // reorder sub-symbols
    for (std::pair<const uint64_t, BinaryData *> &SubBD :
         BC.getSubBinaryData(BD)) {
      if (!SubBD.second->isJumpTable()) {
        uint64_t SubOffset =
            Offset + SubBD.second->getAddress() - BD->getAddress();
        LLVM_DEBUG(dbgs() << "BOLT-DEBUG: SubBD " << SubBD.second->getName()
                          << " @ " << SubOffset << "\n");
        SubBD.second->setOutputLocation(OutputSection, SubOffset);
      }
    }

    Offset += BD->getSize();
    Count += Begin->second;
    NewOrder.push_back(BD);
  }

  OutputSection.reorderContents(NewOrder, opts::ReorderInplace);

  outs() << "BOLT-INFO: reorder-data: " << Count << "/" << TotalCount
         << format(" (%.1f%%)", 100.0 * Count / TotalCount) << " events, "
         << Offset << " hot bytes\n";
}

bool ReorderData::markUnmoveableSymbols(BinaryContext &BC,
                                        BinarySection &Section) const {
  // Private symbols currently can't be moved because data can "leak" across
  // the boundary of one symbol to the next, e.g. a string that has a common
  // suffix might start in one private symbol and end with the common
  // suffix in another.
  auto isPrivate = [&](const BinaryData *BD) {
    auto Prefix = std::string("PG") + BC.AsmInfo->getPrivateGlobalPrefix();
    return BD->getName().startswith(Prefix.str());
  };
  auto Range = BC.getBinaryDataForSection(Section);
  bool FoundUnmoveable = false;
  for (auto Itr = Range.begin(); Itr != Range.end(); ++Itr) {
    if (Itr->second->getName().startswith("PG.")) {
      BinaryData *Prev =
          Itr != Range.begin() ? std::prev(Itr)->second : nullptr;
      BinaryData *Next = Itr != Range.end() ? std::next(Itr)->second : nullptr;
      bool PrevIsPrivate = Prev && isPrivate(Prev);
      bool NextIsPrivate = Next && isPrivate(Next);
      if (isPrivate(Itr->second) && (PrevIsPrivate || NextIsPrivate))
        Itr->second->setIsMoveable(false);
    } else {
      // check for overlapping symbols.
      BinaryData *Next = Itr != Range.end() ? std::next(Itr)->second : nullptr;
      if (Next && Itr->second->getEndAddress() != Next->getAddress() &&
          Next->containsAddress(Itr->second->getEndAddress())) {
        Itr->second->setIsMoveable(false);
        Next->setIsMoveable(false);
      }
    }
    FoundUnmoveable |= !Itr->second->isMoveable();
  }
  return FoundUnmoveable;
}

void ReorderData::runOnFunctions(BinaryContext &BC) {
  static const char *DefaultSections[] = {".rodata", ".data", ".bss", nullptr};

  if (!BC.HasRelocations || opts::ReorderData.empty())
    return;

  // For now
  if (opts::JumpTables > JTS_BASIC) {
    outs() << "BOLT-WARNING: jump table support must be basic for "
           << "data reordering to work.\n";
    return;
  }

  assignMemData(BC);

  std::vector<BinarySection *> Sections;

  for (const std::string &SectionName : opts::ReorderData) {
    if (SectionName == "default") {
      for (unsigned I = 0; DefaultSections[I]; ++I)
        if (ErrorOr<BinarySection &> Section =
                BC.getUniqueSectionByName(DefaultSections[I]))
          Sections.push_back(&*Section);
      continue;
    }

    ErrorOr<BinarySection &> Section = BC.getUniqueSectionByName(SectionName);
    if (!Section) {
      outs() << "BOLT-WARNING: Section " << SectionName
             << " not found, skipping.\n";
      continue;
    }

    if (!isSupported(*Section)) {
      outs() << "BOLT-ERROR: Section " << SectionName << " not supported.\n";
      exit(1);
    }

    Sections.push_back(&*Section);
  }

  for (BinarySection *Section : Sections) {
    const bool FoundUnmoveable = markUnmoveableSymbols(BC, *Section);

    DataOrder Order;
    unsigned SplitPointIdx;

    if (opts::ReorderAlgorithm == opts::ReorderAlgo::REORDER_COUNT) {
      outs() << "BOLT-INFO: reorder-sections: ordering data by count\n";
      std::tie(Order, SplitPointIdx) = sortedByCount(BC, *Section);
    } else {
      outs() << "BOLT-INFO: reorder-sections: ordering data by funcs\n";
      std::tie(Order, SplitPointIdx) =
          sortedByFunc(BC, *Section, BC.getBinaryFunctions());
    }
    auto SplitPoint = Order.begin() + SplitPointIdx;

    if (opts::PrintReorderedData)
      printOrder(*Section, Order.begin(), SplitPoint);

    if (!opts::ReorderInplace || FoundUnmoveable) {
      if (opts::ReorderInplace && FoundUnmoveable)
        outs() << "BOLT-INFO: Found unmoveable symbols in "
               << Section->getName() << " falling back to splitting "
               << "instead of in-place reordering.\n";

      // Rename sections.
      BinarySection &Hot =
          BC.registerSection(Section->getName() + ".hot", *Section);
      Hot.setOutputName(Section->getName());
      Section->setOutputName(".bolt.org" + Section->getName());

      // Reorder contents of original section.
      setSectionOrder(BC, Hot, Order.begin(), SplitPoint);

      // This keeps the original data from thinking it has been moved.
      for (std::pair<const uint64_t, BinaryData *> &Entry :
           BC.getBinaryDataForSection(*Section)) {
        if (!Entry.second->isMoved()) {
          Entry.second->setSection(*Section);
          Entry.second->setOutputSection(*Section);
        }
      }
    } else {
      outs() << "BOLT-WARNING: Inplace section reordering not supported yet.\n";
      setSectionOrder(BC, *Section, Order.begin(), Order.end());
    }
  }
}

} // namespace bolt
} // namespace llvm
