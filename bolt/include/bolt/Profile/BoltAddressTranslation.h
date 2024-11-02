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
  using MapTy = std::map<uint32_t, uint32_t>;

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
  void dump(raw_ostream &OS);

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
  uint64_t fetchParentAddress(uint64_t Address) const;

  /// True if the input binary has a translation table we can use to convert
  /// addresses when aggregating profile
  bool enabledFor(llvm::object::ELFObjectFileBase *InputFile) const;

  /// Save function and basic block hashes used for metadata dump.
  void saveMetadata(BinaryContext &BC);

private:
  /// Helper to update \p Map by inserting one or more BAT entries reflecting
  /// \p BB for function located at \p FuncAddress. At least one entry will be
  /// emitted for the start of the BB. More entries may be emitted to cover
  /// the location of calls or any instruction that may change control flow.
  void writeEntriesForBB(MapTy &Map, const BinaryBasicBlock &BB,
                         uint64_t FuncAddress);

  /// Write the serialized address translation table for a function.
  template <bool Cold>
  void writeMaps(std::map<uint64_t, MapTy> &Maps, uint64_t &PrevAddress,
                 raw_ostream &OS);

  /// Read the serialized address translation table for a function.
  /// Return a parse error if failed.
  template <bool Cold>
  void parseMaps(std::vector<uint64_t> &HotFuncs, uint64_t &PrevAddress,
                 DataExtractor &DE, uint64_t &Offset, Error &Err);

  /// Returns the bitmask with set bits corresponding to indices of BRANCHENTRY
  /// entries in function address translation map.
  APInt calculateBranchEntriesBitMask(MapTy &Map, size_t EqualElems);

  /// Calculate the number of equal offsets (output = input) in the beginning
  /// of the function.
  size_t getNumEqualOffsets(const MapTy &Map) const;

  std::map<uint64_t, MapTy> Maps;

  using BBHashMap = std::unordered_map<uint32_t, size_t>;
  std::unordered_map<uint64_t, std::pair<size_t, BBHashMap>> FuncHashes;

  /// Links outlined cold bocks to their original function
  std::map<uint64_t, uint64_t> ColdPartSource;

  /// Identifies the address of a control-flow changing instructions in a
  /// translation map entry
  const static uint32_t BRANCHENTRY = 0x1;
};
} // namespace bolt

} // namespace llvm

#endif
