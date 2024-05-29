//===- bolt/Core/AddressMap.h - Input-output address map --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the AddressMap class used for looking
// up addresses in the output object.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_ADDRESS_MAP_H
#define BOLT_CORE_ADDRESS_MAP_H

#include "llvm/MC/MCSymbol.h"

#include <optional>
#include <unordered_map>

namespace llvm {

class MCStreamer;

namespace bolt {

class BinaryContext;

/// Helper class to create a mapping from input entities to output addresses
/// needed for updating debugging symbols and BAT. We emit a section containing
/// <Input entity, Output MCSymbol> pairs to the object file and JITLink will
/// transform this in <Input entity, Output address> pairs. The linker output
/// can then be parsed and used to establish the mapping.
///
/// The entities that can be mapped to output address are input addresses and
/// labels (MCSymbol). Input addresses support one-to-many mapping.
class AddressMap {
  static const char *const AddressSectionName;
  static const char *const LabelSectionName;

  /// Map multiple <input address> to <output address>.
  using Addr2AddrMapTy = std::unordered_multimap<uint64_t, uint64_t>;
  Addr2AddrMapTy Address2AddressMap;

  /// Map MCSymbol to its output address. Normally used for temp symbols that
  /// are not updated by the linker.
  using Label2AddrMapTy = DenseMap<const MCSymbol *, uint64_t>;
  Label2AddrMapTy Label2AddrMap;

public:
  static void emit(MCStreamer &Streamer, BinaryContext &BC);
  static std::optional<AddressMap> parse(BinaryContext &BC);

  std::optional<uint64_t> lookup(uint64_t InputAddress) const {
    auto It = Address2AddressMap.find(InputAddress);
    if (It != Address2AddressMap.end())
      return It->second;
    return std::nullopt;
  }

  std::optional<uint64_t> lookup(const MCSymbol *Symbol) const {
    auto It = Label2AddrMap.find(Symbol);
    if (It != Label2AddrMap.end())
      return It->second;
    return std::nullopt;
  }

  std::pair<Addr2AddrMapTy::const_iterator, Addr2AddrMapTy::const_iterator>
  lookupAll(uint64_t InputAddress) const {
    return Address2AddressMap.equal_range(InputAddress);
  }
};

} // namespace bolt
} // namespace llvm

#endif
