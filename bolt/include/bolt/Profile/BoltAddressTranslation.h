//===- bolt/Profile/BoltAddressTranslation.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PROFILE_BOLTADDRESSTRANSLATION_H
#define BOLT_PROFILE_BOLTADDRESSTRANSLATION_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataExtractor.h"
#include <cstdint>
#include <map>
#include <optional>
#include <system_error>
#include <unordered_map>

namespace llvm {
class MCSymbol;
class raw_ostream;

namespace object {
class ELFObjectFileBase;
} // namespace object

namespace bolt {
class BinaryBasicBlock;
class BinaryContext;
class BinaryFunction;

/// The map of output addresses to input ones to be used when translating
/// samples collected in a binary that was already processed by BOLT. We do not
/// support reoptimizing a binary already processed by BOLT, but we do support
/// collecting samples in a binary processed by BOLT. We then translate samples
/// back to addresses from the input (original) binary, one that can be
/// optimized. The goal is to avoid special deployments of non-bolted binaries
/// just for the purposes of data collection.
///
/// The in-memory representation of the map is as follows. Each function has its
/// own map. A function is identified by its output address. This is the key to
/// retrieve a translation map. The translation map is a collection of ordered
/// keys identifying the start of a region (relative to the function start) in
/// the output address space (addresses in the binary processed by BOLT).
///
/// A translation then happens when perf2bolt needs to convert sample addresses
/// in the output address space back to input addresses, valid to run BOLT in
/// the original input binary. To convert, perf2bolt first needs to fetch the
/// translation map for a sample recorded in a given function. It then finds
/// the largest key that is still smaller or equal than the recorded address.
/// It then converts this address to use the value of this key.
///
///   Example translation Map for function foo
///      KEY                             VALUE                    BB?
///    Output offset1 (first BB)         Original input offset1   Y
///    ...
///    Output offsetN (last branch)      Original input offsetN   N
///
/// The information on whether a given entry is a BB start or an instruction
/// that changes control flow is encoded in the last (highest) bit of VALUE.
///
/// Notes:
/// Instructions that will never appear in LBR because they do not cause control
/// flow change are omitted from this map. Basic block locations are recorded
/// because they can be a target of a jump (To address in the LBR) and also to
/// recreate the BB layout of this function. We use the BB layout map to
/// recreate fall-through jumps in the profile, given an LBR trace.
class BoltAddressTranslation {
public:
  // In-memory representation of the address translation table
  using MapTy = std::multimap<uint32_t, uint32_t>;

  // List of taken fall-throughs
  using FallthroughListTy = SmallVector<std::pair<uint64_t, uint64_t>, 16>;

  /// Name of the ELF section where the table will be serialized to in the
  /// output binary
  static const char *SECTION_NAME;

  BoltAddressTranslation() {}

  /// Write the serialized address translation tables for each reordered
  /// function
  void write(const BinaryContext &BC, raw_ostream &OS);

  /// Read the serialized address translation tables and load them internally
  /// in memory. Return a parse error if failed.
  std::error_code parse(raw_ostream &OS, StringRef Buf);

  /// Dump the parsed address translation tables
  void dump(raw_ostream &OS) const;

  /// If the maps are loaded in memory, perform the lookup to translate LBR
  /// addresses in function located at \p FuncAddress.
  uint64_t translate(uint64_t FuncAddress, uint64_t Offset,
                     bool IsBranchSrc) const;

  /// Use the map keys containing basic block addresses to infer fall-throughs
  /// taken in the path started at FirstLBR.To and ending at SecondLBR.From.
  /// Return std::nullopt if trace is invalid or the list of fall-throughs
  /// otherwise.
  std::optional<FallthroughListTy> getFallthroughsInTrace(uint64_t FuncAddress,
                                                          uint64_t From,
                                                          uint64_t To) const;

  /// If available, fetch the address of the hot part linked to the cold part
  /// at \p Address. Return 0 otherwise.
  uint64_t fetchParentAddress(uint64_t Address) const {
    auto Iter = ColdPartSource.find(Address);
    if (Iter == ColdPartSource.end())
      return 0;
    return Iter->second;
  }

  /// True if the input binary has a translation table we can use to convert
  /// addresses when aggregating profile
  bool enabledFor(llvm::object::ELFObjectFileBase *InputFile) const;

  /// Save function and basic block hashes used for metadata dump.
  void saveMetadata(BinaryContext &BC);

  /// True if a given \p Address is a function with translation table entry.
  bool isBATFunction(uint64_t Address) const { return Maps.count(Address); }

  /// For a given \p Symbol in the output binary and known \p InputOffset
  /// return a corresponding pair of parent BinaryFunction and secondary entry
  /// point in it.
  std::pair<const BinaryFunction *, unsigned>
  translateSymbol(const BinaryContext &BC, const MCSymbol &Symbol,
                  uint32_t InputOffset) const;

private:
  /// Helper to update \p Map by inserting one or more BAT entries reflecting
  /// \p BB for function located at \p FuncAddress. At least one entry will be
  /// emitted for the start of the BB. More entries may be emitted to cover
  /// the location of calls or any instruction that may change control flow.
  void writeEntriesForBB(MapTy &Map, const BinaryBasicBlock &BB,
                         uint64_t FuncInputAddress,
                         uint64_t FuncOutputAddress) const;

  /// Write the serialized address translation table for a function.
  template <bool Cold> void writeMaps(uint64_t &PrevAddress, raw_ostream &OS);

  /// Read the serialized address translation table for a function.
  /// Return a parse error if failed.
  template <bool Cold>
  void parseMaps(uint64_t &PrevAddress, DataExtractor &DE, uint64_t &Offset,
                 Error &Err);

  /// Returns the bitmask with set bits corresponding to indices of BRANCHENTRY
  /// entries in function address translation map.
  APInt calculateBranchEntriesBitMask(MapTy &Map, size_t EqualElems) const;

  /// Calculate the number of equal offsets (output = input - skew) in the
  /// beginning of the function.
  size_t getNumEqualOffsets(const MapTy &Map, uint32_t Skew) const;

  std::map<uint64_t, MapTy> Maps;

  /// Ordered vector with addresses of hot functions.
  std::vector<uint64_t> HotFuncs;

  /// Map a function to its basic blocks count
  std::unordered_map<uint64_t, size_t> NumBasicBlocksMap;

  /// Map a function to its secondary entry points vector
  std::unordered_map<uint64_t, std::vector<uint32_t>> SecondaryEntryPointsMap;

  /// Return a secondary entry point ID for a function located at \p Address and
  /// \p Offset within that function.
  unsigned getSecondaryEntryPointId(uint64_t Address, uint32_t Offset) const;

  /// Links outlined cold bocks to their original function
  std::map<uint64_t, uint64_t> ColdPartSource;

  /// Links output address of a main fragment back to input address.
  std::unordered_map<uint64_t, uint64_t> ReverseMap;

  /// Identifies the address of a control-flow changing instructions in a
  /// translation map entry
  const static uint32_t BRANCHENTRY = 0x1;

public:
  /// Map basic block input offset to a basic block index and hash pair.
  class BBHashMapTy {
    struct EntryTy {
      unsigned Index;
      size_t Hash;
    };

    std::map<uint32_t, EntryTy> Map;
    const EntryTy &getEntry(uint32_t BBInputOffset) const {
      auto It = Map.find(BBInputOffset);
      assert(It != Map.end());
      return It->second;
    }

  public:
    bool isInputBlock(uint32_t InputOffset) const {
      return Map.count(InputOffset);
    }

    unsigned getBBIndex(uint32_t BBInputOffset) const {
      return getEntry(BBInputOffset).Index;
    }

    size_t getBBHash(uint32_t BBInputOffset) const {
      return getEntry(BBInputOffset).Hash;
    }

    void addEntry(uint32_t BBInputOffset, unsigned BBIndex, size_t BBHash) {
      Map.emplace(BBInputOffset, EntryTy{BBIndex, BBHash});
    }

    size_t getNumBasicBlocks() const { return Map.size(); }

    auto begin() const { return Map.begin(); }
    auto end() const { return Map.end(); }
    auto upper_bound(uint32_t Offset) const { return Map.upper_bound(Offset); }
    auto size() const { return Map.size(); }
  };

  /// Map function output address to its hash and basic blocks hash map.
  class FuncHashesTy {
    struct EntryTy {
      size_t Hash;
      BBHashMapTy BBHashMap;
    };

    std::unordered_map<uint64_t, EntryTy> Map;
    const EntryTy &getEntry(uint64_t FuncOutputAddress) const {
      auto It = Map.find(FuncOutputAddress);
      assert(It != Map.end());
      return It->second;
    }

  public:
    size_t getBFHash(uint64_t FuncOutputAddress) const {
      return getEntry(FuncOutputAddress).Hash;
    }

    const BBHashMapTy &getBBHashMap(uint64_t FuncOutputAddress) const {
      return getEntry(FuncOutputAddress).BBHashMap;
    }

    void addEntry(uint64_t FuncOutputAddress, size_t BFHash) {
      Map.emplace(FuncOutputAddress, EntryTy{BFHash, BBHashMapTy()});
    }

    size_t getNumFunctions() const { return Map.size(); };

    size_t getNumBasicBlocks() const {
      size_t NumBasicBlocks{0};
      for (auto &I : Map)
        NumBasicBlocks += I.second.BBHashMap.getNumBasicBlocks();
      return NumBasicBlocks;
    }
  };

  /// Returns BF hash by function output address (after BOLT).
  size_t getBFHash(uint64_t FuncOutputAddress) const {
    return FuncHashes.getBFHash(FuncOutputAddress);
  }

  /// Returns BBHashMap by function output address (after BOLT).
  const BBHashMapTy &getBBHashMap(uint64_t FuncOutputAddress) const {
    return FuncHashes.getBBHashMap(FuncOutputAddress);
  }

  BBHashMapTy &getBBHashMap(uint64_t FuncOutputAddress) {
    return const_cast<BBHashMapTy &>(
        std::as_const(*this).getBBHashMap(FuncOutputAddress));
  }

  /// Returns the number of basic blocks in a function.
  size_t getNumBasicBlocks(uint64_t OutputAddress) const {
    auto It = NumBasicBlocksMap.find(OutputAddress);
    assert(It != NumBasicBlocksMap.end());
    return It->second;
  }

private:
  FuncHashesTy FuncHashes;
};
} // namespace bolt

} // namespace llvm

#endif
