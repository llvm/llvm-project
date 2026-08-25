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
#include <memory>

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

/// Target-specific relocation operations. One handler is owned by each
/// BinaryContext, while Relocation remains a lightweight value type.
class RelocationHandler {
public:
  virtual ~RelocationHandler() = default;

  /// Check if \p Type is a supported relocation type.
  virtual bool isSupported(uint32_t Type) const = 0;

  /// Return size in bytes of the given relocation \p Type.
  virtual size_t getSizeForType(uint32_t Type) const = 0;

  /// Skip relocations that we don't want to handle in BOLT
  virtual bool skipRelocationType(uint32_t Type) const = 0;

  /// Adjust value depending on relocation type (make it PC relative or not).
  virtual uint64_t encodeValue(uint32_t Type, uint64_t Value,
                               uint64_t PC) const = 0;

  /// Return true if there are enough bits to encode the relocation value.
  virtual bool canEncodeValue(uint32_t Type, uint64_t Value,
                              uint64_t PC) const = 0;

  /// Extract current relocated value from binary contents. This is used for
  /// RISC architectures where values are encoded in specific bits depending
  /// on the relocation value. For X86, we limit to sign extending the value
  /// if necessary.
  virtual uint64_t extractValue(uint32_t Type, uint64_t Contents,
                                uint64_t PC) const = 0;

  /// Return true if relocation type implies the creation of a GOT entry
  virtual bool isGOT(uint32_t Type) const = 0;

  /// Return true if relocation type is NONE
  bool isNone(uint32_t Type) const { return Type == getNone(); }

  /// Return true if relocation type is RELATIVE
  virtual bool isRelative(uint32_t Type) const = 0;

  /// Return true if relocation type is IRELATIVE
  virtual bool isIRelative(uint32_t Type) const = 0;

  /// Return true if relocation type is for thread local storage.
  virtual bool isTLS(uint32_t Type) const = 0;

  /// Return true of relocation type is for referencing a specific instruction
  /// (as opposed to a function, basic block, etc).
  virtual bool isInstructionReference(uint32_t Type) const { return false; }

  /// Return code for a NONE relocation
  virtual uint32_t getNone() const = 0;

  /// Return code for a PC-relative 4-byte relocation
  virtual uint32_t getPC32() const = 0;

  /// Return code for a PC-relative 8-byte relocation
  virtual uint32_t getPC64() const = 0;

  /// Return true if relocation type is PC-relative. Return false otherwise.
  virtual bool isPCRelative(uint32_t Type) const = 0;

  /// Return code for a ABS 8-byte relocation
  virtual uint32_t getAbs64() const = 0;

  /// Return code for a RELATIVE relocation
  virtual uint32_t getRelative() const = 0;
  virtual MCBinaryExpr::Opcode getComposeOpcodeFor(uint32_t Type) const;
  virtual void printType(raw_ostream &OS, uint32_t Type) const = 0;
};

std::unique_ptr<RelocationHandler>
createRelocationHandler(Triple::ArchType Arch);

/// Relocation class.
class Relocation {
public:
  Relocation(uint64_t Offset, MCSymbol *Symbol, uint32_t Type, uint64_t Addend,
             uint64_t Value, bool IsRELR = false)
      : Offset(Offset), Symbol(Symbol), Type(Type), Optional(false),
        IsRELR(IsRELR), Addend(Addend), Value(Value) {}

  Relocation()
      : Offset(0), Symbol(0), Type(0), Optional(0), IsRELR(0), Addend(0),
        Value(0) {}

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

  /// Track which relocations originate from a relr section. Emit these
  /// exclusively into the relr section and do not accidentally promote relative
  /// rela entries, because that would require growing the relr section.
  bool IsRELR = false;

public:
  /// The offset from the \p Symbol base used to compute the final
  /// value of this relocation.
  uint64_t Addend;

  /// The computed relocation value extracted from the binary file.
  /// Used to validate relocation correctness.
  uint64_t Value;

  void setOptional() { Optional = true; }

  bool isOptional() { return Optional; }

  bool isRELR() const { return IsRELR; }

  /// Return size of this relocation.
  /// Return the relocation type of \p Rel from llvm::object. It checks for
  /// overflows as BOLT uses 32 bits for the type.
  static uint32_t getType(const object::RelocationRef &Rel);

  size_t getSize(const RelocationHandler &RH) const { return RH.getSizeForType(Type);
  }

  /// Emit relocation at a current \p Streamer' position. The caller is
  /// responsible for setting the position correctly.
  size_t emit(MCStreamer *Streamer, const RelocationHandler &RH) const;

  /// Emit a group of composed relocations. All relocations must have the same
  /// offset. If std::distance(Begin, End) == 1, this is equivalent to
  /// Begin->emit(Streamer).
  template <typename RelocIt>
  static size_t emit(RelocIt Begin, RelocIt End, MCStreamer *Streamer,
                     const RelocationHandler &RH) {
    if (Begin == End)
      return 0;

    const MCExpr *Value = nullptr;

    for (auto RI = Begin; RI != End; ++RI) {
      assert(RI->Offset == Begin->Offset &&
             "emitting composed relocations with different offsets");
      Value = RI->createExpr(Streamer, Value, RH);
    }

    assert(Value && "failed to create relocation value");
    auto Size = std::prev(End)->getSize(RH);
    Streamer->emitValue(Value, Size);
    return Size;
  }

  /// Print a relocation to \p OS.
  void print(raw_ostream &OS, const RelocationHandler &RH) const;

private:
  const MCExpr *createExpr(MCStreamer *Streamer,
                           const RelocationHandler &RH) const;
  const MCExpr *createExpr(MCStreamer *Streamer,
                           const MCExpr *RetainedValue,
                           const RelocationHandler &RH) const;
};

/// Relocation ordering by offset.
inline bool operator<(const Relocation &A, const Relocation &B) {
  return A.Offset < B.Offset;
}

inline bool operator<(const Relocation &A, uint64_t B) { return A.Offset < B; }

inline bool operator<(uint64_t A, const Relocation &B) { return A < B.Offset; }

} // namespace bolt
} // namespace llvm

#endif
