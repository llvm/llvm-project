//===- bolt/Core/Linker.h - BOLTLinker interface ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the interface BOLT uses for linking.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_LINKER_H
#define BOLT_CORE_LINKER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <cstdint>
#include <functional>
#include <optional>

namespace llvm {
namespace bolt {

class BinarySection;

class BOLTLinker {
public:
  using SectionMapper =
      std::function<void(const BinarySection &Section, uint64_t Address)>;
  using SectionsMapper = std::function<void(SectionMapper)>;

  struct SymbolInfo {
    uint64_t Address;
    uint64_t Size;
  };

  virtual ~BOLTLinker() = default;

  /// Load and link \p Obj. \p MapSections will be called before the object is
  /// linked to allow section addresses to be remapped. When called, the address
  /// of a section can be changed by calling the passed SectionMapper.
  virtual void loadObject(MemoryBufferRef Obj, SectionsMapper MapSections) = 0;

  /// Return the address and size of a symbol or std::nullopt if it cannot be
  /// found.
  virtual std::optional<SymbolInfo> lookupSymbolInfo(StringRef Name) const = 0;

  /// Return the address of a symbol or std::nullopt if it cannot be found.
  std::optional<uint64_t> lookupSymbol(StringRef Name) const {
    if (const auto Info = lookupSymbolInfo(Name))
      return Info->Address;
    return std::nullopt;
  }
};

} // namespace bolt
} // namespace llvm

#endif // BOLT_CORE_LINKER_H
