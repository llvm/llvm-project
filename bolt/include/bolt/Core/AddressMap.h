//===- bolt/Core/AddressMap.h - Input-output address map --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper class to create a mapping from input to output addresses needed for
// updating debugging symbols and BAT. We emit an MCSection containing
// <Input address, Output MCSymbol> pairs to the object file and JITLink will
// transform this in <Input address, Output address> pairs. The linker output
// can then be parsed and used to establish the mapping.
//
//===----------------------------------------------------------------------===//
//
#ifndef BOLT_CORE_ADDRESS_MAP_H
#define BOLT_CORE_ADDRESS_MAP_H

#include "llvm/ADT/StringRef.h"

#include <optional>
#include <unordered_map>

namespace llvm {

class MCStreamer;

namespace bolt {

class BinaryContext;

class AddressMap {
  using MapTy = std::unordered_multimap<uint64_t, uint64_t>;
  MapTy Map;

public:
  static const char *const SectionName;

  static void emit(MCStreamer &Streamer, BinaryContext &BC);
  static AddressMap parse(StringRef Buffer, const BinaryContext &BC);

  std::optional<uint64_t> lookup(uint64_t InputAddress) const {
    auto It = Map.find(InputAddress);
    if (It != Map.end())
      return It->second;
    return std::nullopt;
  }

  std::pair<MapTy::const_iterator, MapTy::const_iterator>
  lookupAll(uint64_t InputAddress) const {
    return Map.equal_range(InputAddress);
  }
};

} // namespace bolt
} // namespace llvm

#endif
