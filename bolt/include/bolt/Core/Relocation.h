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

#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
class MCSymbol;

namespace object {
class RelocationRef;
} // namespace object

class raw_ostream;

namespace ELF {
/// Relocation type mask that was accidentally output by bfd 2.30 linker.
enum { R_X86_64_converted_reloc_bit = 0x80 };
} // namespace ELF

namespace bolt {

/// Relocation class.
class Relocation {
public:
  Relocation(uint64_t Offset, MCSymbol *Symbol, uint32_t Type, uint64_t Addend,
             uint64_t Value)
      : Offset(Offset), Symbol(Symbol), Type(Type), Optional(false),
        Addend(Addend), Value(Value) {}

  Relocation()
      : Offset(0), Symbol(0), Type(0), Optional(0), Addend(0), Value(0) {}

  static Triple::ArchType Arch; /// set by BinaryContext ctor.

  /// The offset of this relocation in the object it is contained in.
  uint64_t Offset;

  /// The symbol this relocation is referring to.
  MCSymbol *Symbol;

  /// Relocation type.
  uint32_t Type;

private:
  /// Relocations added by optimizations can be optional, meaning they can be
  /// omitted under certain circumstances.
  bool Optional = false;

public:
  /// The offset from the \p Symbol base used to compute the final
  /// value of this relocation.
  uint64_t Addend;

  /// The computed relocation value extracted from the binary file.
  /// Used to validate relocation correctness.
  uint64_t Value;

  /// Return size in bytes of the given relocation \p Type.
  static size_t getSizeForType(uint32_t Type);

  void setOptional() { Optional = true; }

  bool isOptional() { return Optional; }

  /// Return size of this relocation.
  size_t getSize() const { return getSizeForType(Type); }

  /// Skip relocations that we don't want to handle in BOLT
  static bool skipRelocationType(uint32_t Type);

  /// Adjust value depending on relocation type (make it PC relative or not).
  static uint64_t encodeValue(uint32_t Type, uint64_t Value, uint64_t PC);

  /// Return true if there are enough bits to encode the relocation value.
  static bool canEncodeValue(uint32_t Type, uint64_t Value, uint64_t PC);

  /// Extract current relocated value from binary contents. This is used for
  /// RISC architectures where values are encoded in specific bits depending
  /// on the relocation value. For X86, we limit to sign extending the value
  /// if necessary.
  static uint64_t extractValue(uint32_t Type, uint64_t Contents, uint64_t PC);

  /// Return true if relocation type is PC-relative. Return false otherwise.
  static bool isPCRelative(uint32_t Type);

  /// Check if \p Type is a supported relocation type.
  static bool isSupported(uint32_t Type);

  /// Return true if relocation type implies the creation of a GOT entry
  static bool isGOT(uint32_t Type);

  /// Special relocation type that allows the linker to modify the instruction.
  static bool isX86GOTPCRELX(uint32_t Type);
  static bool isX86GOTPC64(uint32_t Type);

  /// Return true if relocation type is NONE
  static bool isNone(uint32_t Type);

  /// Return true if relocation type is RELATIVE
  static bool isRelative(uint32_t Type);

  /// Return true if relocation type is IRELATIVE
  static bool isIRelative(uint32_t Type);

  /// Return true if relocation type is for thread local storage.
  static bool isTLS(uint32_t Type);

  /// Return true of relocation type is for referencing a specific instruction
  /// (as opposed to a function, basic block, etc).
  static bool isInstructionReference(uint32_t Type);

  /// Return the relocation type of \p Rel from llvm::object. It checks for
  /// overflows as BOLT uses 32 bits for the type.
  static uint32_t getType(const object::RelocationRef &Rel);

  /// Return code for a NONE relocation
  static uint32_t getNone();

  /// Return code for a PC-relative 4-byte relocation
  static uint32_t getPC32();

  /// Return code for a PC-relative 8-byte relocation
  static uint32_t getPC64();

  /// Return code for a ABS 8-byte relocation
  static uint32_t getAbs64();

  /// Return code for a RELATIVE relocation
  static uint32_t getRelative();

  /// Return true if this relocation is PC-relative. Return false otherwise.
  bool isPCRelative() const { return isPCRelative(Type); }

  /// Return true if this relocation is R_*_RELATIVE type. Return false
  /// otherwise.
  bool isRelative() const { return isRelative(Type); }

  /// Return true if this relocation is R_*_IRELATIVE type. Return false
  /// otherwise.
  bool isIRelative() const { return isIRelative(Type); }

  /// Emit relocation at a current \p Streamer' position. The caller is
  /// responsible for setting the position correctly.
  size_t emit(MCStreamer *Streamer) const;

  /// Emit a group of composed relocations. All relocations must have the same
  /// offset. If std::distance(Begin, End) == 1, this is equivalent to
  /// Begin->emit(Streamer).
  template <typename RelocIt>
  static size_t emit(RelocIt Begin, RelocIt End, MCStreamer *Streamer) {
    if (Begin == End)
      return 0;

    const MCExpr *Value = nullptr;

    for (auto RI = Begin; RI != End; ++RI) {
      assert(RI->Offset == Begin->Offset &&
             "emitting composed relocations with different offsets");
      Value = RI->createExpr(Streamer, Value);
    }

    assert(Value && "failed to create relocation value");
    auto Size = std::prev(End)->getSize();
    Streamer->emitValue(Value, Size);
    return Size;
  }

  /// Print a relocation to \p OS.
  void print(raw_ostream &OS) const;

private:
  const MCExpr *createExpr(MCStreamer *Streamer) const;
  const MCExpr *createExpr(MCStreamer *Streamer,
                           const MCExpr *RetainedValue) const;
  static MCBinaryExpr::Opcode getComposeOpcodeFor(uint32_t Type);
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
