//===- bolt/Core/Relocation.h - Object file relocations ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of Relocation class, which represents a
// relocation in an object or a binary file.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_RELOCATION_H
#define BOLT_CORE_RELOCATION_H

#include "llvm/ADT/Triple.h"

namespace llvm {
class MCStreamer;
class MCSymbol;
class raw_ostream;

namespace ELF {
/// Relocation type mask that was accidentally output by bfd 2.30 linker.
enum { R_X86_64_converted_reloc_bit = 0x80 };
} // namespace ELF

namespace bolt {

/// Relocation class.
struct Relocation {
  static Triple::ArchType Arch; /// set by BinaryContext ctor.

  /// The offset of this relocation in the object it is contained in.
  uint64_t Offset;

  /// The symbol this relocation is referring to.
  MCSymbol *Symbol;

  /// Relocation type.
  uint64_t Type;

  /// The offset from the \p Symbol base used to compute the final
  /// value of this relocation.
  uint64_t Addend;

  /// The computed relocation value extracted from the binary file.
  /// Used to validate relocation correctness.
  uint64_t Value;

  /// Return size in bytes of the given relocation \p Type.
  static size_t getSizeForType(uint64_t Type);

  /// Return size of this relocation.
  size_t getSize() const { return getSizeForType(Type); }

  /// Skip relocations that we don't want to handle in BOLT
  static bool skipRelocationType(uint64_t Type);

  /// Handle special cases when relocation should not be processed by BOLT
  static bool skipRelocationProcess(uint64_t Type, uint64_t Contents);

  // Adjust value depending on relocation type (make it PC relative or not)
  static uint64_t adjustValue(uint64_t Type, uint64_t Value,
                              uint64_t PC);

  /// Extract current relocated value from binary contents. This is used for
  /// RISC architectures where values are encoded in specific bits depending
  /// on the relocation value. For X86, we limit to sign extending the value
  /// if necessary.
  static uint64_t extractValue(uint64_t Type, uint64_t Contents, uint64_t PC);

  /// Return true if relocation type is PC-relative. Return false otherwise.
  static bool isPCRelative(uint64_t Type);

  /// Check if \p Type is a supported relocation type.
  static bool isSupported(uint64_t Type);

  /// Return true if relocation type implies the creation of a GOT entry
  static bool isGOT(uint64_t Type);

  /// Special relocation type that allows the linker to modify the instruction.
  static bool isX86GOTPCRELX(uint64_t Type);

  /// Return true if relocation type is NONE
  static bool isNone(uint64_t Type);

  /// Return true if relocation type is RELATIVE
  static bool isRelative(uint64_t Type);

  /// Return true if relocation type is IRELATIVE
  static bool isIRelative(uint64_t Type);

  /// Return true if relocation type is for thread local storage.
  static bool isTLS(uint64_t Type);

  /// Return code for a NONE relocation
  static uint64_t getNone();

  /// Return code for a PC-relative 4-byte relocation
  static uint64_t getPC32();

  /// Return code for a PC-relative 8-byte relocation
  static uint64_t getPC64();

  /// Return true if this relocation is PC-relative. Return false otherwise.
  bool isPCRelative() const { return isPCRelative(Type); }

  /// Return true if this relocation is R_*_RELATIVE type. Return false
  /// otherwise.
  bool isRelative() const { return isRelative(Type); }

  /// Emit relocation at a current \p Streamer' position. The caller is
  /// responsible for setting the position correctly.
  size_t emit(MCStreamer *Streamer) const;

  /// Print a relocation to \p OS.
  void print(raw_ostream &OS) const;
};

/// Relocation ordering by offset.
inline bool operator<(const Relocation &A, const Relocation &B) {
  return A.Offset < B.Offset;
}

inline bool operator<(const Relocation &A, uint64_t B) { return A.Offset < B; }

inline bool operator<(uint64_t A, const Relocation &B) { return A < B.Offset; }

inline raw_ostream &operator<<(raw_ostream &OS, const Relocation &Rel) {
  Rel.print(OS);
  return OS;
}

} // namespace bolt
} // namespace llvm

#endif
