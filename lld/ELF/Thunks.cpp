//===- Thunks.cpp --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This file contains Thunk subclasses.
//
// A thunk is a small piece of code written after an input section
// which is used to jump between "incompatible" functions
// such as MIPS PIC and non-PIC or ARM non-Thumb and Thumb functions.
//
// If a jump target is too far and its address doesn't fit to a
// short jump instruction, we need to create a thunk too, but we
// haven't supported it yet.
//
// i386 and x86-64 don't need thunks.
//
//===---------------------------------------------------------------------===//

#include "Thunks.h"
#include "Config.h"
#include "Error.h"
#include "InputSection.h"
#include "Memory.h"
#include "OutputSections.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <cstring>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;

namespace lld {
namespace elf {

namespace {

// Specific ARM Thunk implementations. The naming convention is:
// Source State, TargetState, Target Requirement, ABS or PI, Range
template <class ELFT> class ARMV7ABSLongThunk final : public Thunk {
public:
  ARMV7ABSLongThunk(const SymbolBody &Dest) : Thunk(Dest) {}

  uint32_t size() const override { return 12; }
  void writeTo(uint8_t *Buf, ThunkSection &IS) const override;
  void addSymbols(ThunkSection &IS) override;
};

template <class ELFT> class ARMV7PILongThunk final : public Thunk {
public:
  ARMV7PILongThunk(const SymbolBody &Dest) : Thunk(Dest) {}

  uint32_t size() const override { return 16; }
  void writeTo(uint8_t *Buf, ThunkSection &IS) const override;
  void addSymbols(ThunkSection &IS) override;
};

template <class ELFT> class ThumbV7ABSLongThunk final : public Thunk {
public:
  ThumbV7ABSLongThunk(const SymbolBody &Dest) : Thunk(Dest) {
    this->alignment = 2;
  }

  uint32_t size() const override { return 10; }
  void writeTo(uint8_t *Buf, ThunkSection &IS) const override;
  void addSymbols(ThunkSection &IS) override;
};

template <class ELFT> class ThumbV7PILongThunk final : public Thunk {
public:
  ThumbV7PILongThunk(const SymbolBody &Dest) : Thunk(Dest) {
    this->alignment = 2;
  }

  uint32_t size() const override { return 12; }
  void writeTo(uint8_t *Buf, ThunkSection &IS) const override;
  void addSymbols(ThunkSection &IS) override;
};

// MIPS LA25 thunk
template <class ELFT> class MipsThunk final : public Thunk {
public:
  MipsThunk(const SymbolBody &Dest) : Thunk(Dest) {}

  uint32_t size() const override { return 16; }
  void writeTo(uint8_t *Buf, ThunkSection &IS) const override;
  void addSymbols(ThunkSection &IS) override;
  InputSection *getTargetInputSection() const override;
};

} // end anonymous namespace

// ARM Target Thunks
static uint64_t getARMThunkDestVA(const SymbolBody &S) {
  uint64_t V = S.isInPlt() ? S.getPltVA() : S.getVA();
  return SignExtend64<32>(V);
}

template <class ELFT>
void ARMV7ABSLongThunk<ELFT>::writeTo(uint8_t *Buf, ThunkSection &IS) const {
  const uint8_t Data[] = {
      0x00, 0xc0, 0x00, 0xe3, // movw         ip,:lower16:S
      0x00, 0xc0, 0x40, 0xe3, // movt         ip,:upper16:S
      0x1c, 0xff, 0x2f, 0xe1, // bx   ip
  };
  uint64_t S = getARMThunkDestVA(this->Destination);
  memcpy(Buf, Data, sizeof(Data));
  Target->relocateOne(Buf, R_ARM_MOVW_ABS_NC, S);
  Target->relocateOne(Buf + 4, R_ARM_MOVT_ABS, S);
}

template <class ELFT>
void ARMV7ABSLongThunk<ELFT>::addSymbols(ThunkSection &IS) {
  this->ThunkSym = addSyntheticLocal<ELFT>(
      Saver.save("__ARMv7ABSLongThunk_" + this->Destination.getName()),
      STT_FUNC, this->Offset, size(), &IS);
  addSyntheticLocal<ELFT>("$a", STT_NOTYPE, this->Offset, 0, &IS);
}

template <class ELFT>
void ThumbV7ABSLongThunk<ELFT>::writeTo(uint8_t *Buf, ThunkSection &IS) const {
  const uint8_t Data[] = {
      0x40, 0xf2, 0x00, 0x0c, // movw         ip, :lower16:S
      0xc0, 0xf2, 0x00, 0x0c, // movt         ip, :upper16:S
      0x60, 0x47,             // bx   ip
  };
  uint64_t S = getARMThunkDestVA(this->Destination);
  memcpy(Buf, Data, sizeof(Data));
  Target->relocateOne(Buf, R_ARM_THM_MOVW_ABS_NC, S);
  Target->relocateOne(Buf + 4, R_ARM_THM_MOVT_ABS, S);
}

template <class ELFT>
void ThumbV7ABSLongThunk<ELFT>::addSymbols(ThunkSection &IS) {
  this->ThunkSym = addSyntheticLocal<ELFT>(
      Saver.save("__Thumbv7ABSLongThunk_" + this->Destination.getName()),
      STT_FUNC, this->Offset, size(), &IS);
  addSyntheticLocal<ELFT>("$t", STT_NOTYPE, this->Offset, 0, &IS);
}

template <class ELFT>
void ARMV7PILongThunk<ELFT>::writeTo(uint8_t *Buf, ThunkSection &IS) const {
  const uint8_t Data[] = {
      0xf0, 0xcf, 0x0f, 0xe3, // P:  movw ip,:lower16:S - (P + (L1-P) +8)
      0x00, 0xc0, 0x40, 0xe3, //     movt ip,:upper16:S - (P + (L1-P+4) +8)
      0x0f, 0xc0, 0x8c, 0xe0, // L1: add ip, ip, pc
      0x1c, 0xff, 0x2f, 0xe1, //     bx r12
  };
  uint64_t S = getARMThunkDestVA(this->Destination);
  uint64_t P = this->ThunkSym->getVA();
  memcpy(Buf, Data, sizeof(Data));
  Target->relocateOne(Buf, R_ARM_MOVW_PREL_NC, S - P - 16);
  Target->relocateOne(Buf + 4, R_ARM_MOVT_PREL, S - P - 12);
}

template <class ELFT>
void ARMV7PILongThunk<ELFT>::addSymbols(ThunkSection &IS) {
  this->ThunkSym = addSyntheticLocal<ELFT>(
      Saver.save("__ARMV7PILongThunk_" + this->Destination.getName()), STT_FUNC,
      this->Offset, size(), &IS);
  addSyntheticLocal<ELFT>("$a", STT_NOTYPE, this->Offset, 0, &IS);
}

template <class ELFT>
void ThumbV7PILongThunk<ELFT>::writeTo(uint8_t *Buf, ThunkSection &IS) const {
  const uint8_t Data[] = {
      0x4f, 0xf6, 0xf4, 0x7c, // P:  movw ip,:lower16:S - (P + (L1-P) + 4)
      0xc0, 0xf2, 0x00, 0x0c, //     movt ip,:upper16:S - (P + (L1-P+4) + 4)
      0xfc, 0x44,             // L1: add  r12, pc
      0x60, 0x47,             //     bx   r12
  };
  uint64_t S = getARMThunkDestVA(this->Destination);
  uint64_t P = this->ThunkSym->getVA();
  memcpy(Buf, Data, sizeof(Data));
  Target->relocateOne(Buf, R_ARM_THM_MOVW_PREL_NC, S - P - 12);
  Target->relocateOne(Buf + 4, R_ARM_THM_MOVT_PREL, S - P - 8);
}

template <class ELFT>
void ThumbV7PILongThunk<ELFT>::addSymbols(ThunkSection &IS) {
  this->ThunkSym = addSyntheticLocal<ELFT>(
      Saver.save("__ThumbV7PILongThunk_" + this->Destination.getName()),
      STT_FUNC, this->Offset, size(), &IS);
  addSyntheticLocal<ELFT>("$t", STT_NOTYPE, this->Offset, 0, &IS);
}

// Write MIPS LA25 thunk code to call PIC function from the non-PIC one.
template <class ELFT>
void MipsThunk<ELFT>::writeTo(uint8_t *Buf, ThunkSection &) const {
  const endianness E = ELFT::TargetEndianness;

  uint64_t S = this->Destination.getVA();
  write32<E>(Buf, 0x3c190000);                // lui   $25, %hi(func)
  write32<E>(Buf + 4, 0x08000000 | (S >> 2)); // j     func
  write32<E>(Buf + 8, 0x27390000);            // addiu $25, $25, %lo(func)
  write32<E>(Buf + 12, 0x00000000);           // nop
  Target->relocateOne(Buf, R_MIPS_HI16, S);
  Target->relocateOne(Buf + 8, R_MIPS_LO16, S);
}

template <class ELFT> void MipsThunk<ELFT>::addSymbols(ThunkSection &IS) {
  this->ThunkSym = addSyntheticLocal<ELFT>(
      Saver.save("__LA25Thunk_" + this->Destination.getName()), STT_FUNC,
      this->Offset, size(), &IS);
}

template <class ELFT>
InputSection *MipsThunk<ELFT>::getTargetInputSection() const {
  auto *DR = dyn_cast<DefinedRegular>(&this->Destination);
  return dyn_cast<InputSection>(DR->Section);
}

Thunk::Thunk(const SymbolBody &D) : Destination(D), Offset(0) {}

Thunk::~Thunk() = default;

// Creates a thunk for Thumb-ARM interworking.
template <class ELFT> static Thunk *addThunkArm(uint32_t Reloc, SymbolBody &S) {
  // ARM relocations need ARM to Thumb interworking Thunks.
  // Thumb relocations need Thumb to ARM relocations.
  // Use position independent Thunks if we require position independent code.
  switch (Reloc) {
  case R_ARM_PC24:
  case R_ARM_PLT32:
  case R_ARM_JUMP24:
    if (Config->Pic)
      return make<ARMV7PILongThunk<ELFT>>(S);
    return make<ARMV7ABSLongThunk<ELFT>>(S);
  case R_ARM_THM_JUMP19:
  case R_ARM_THM_JUMP24:
    if (Config->Pic)
      return make<ThumbV7PILongThunk<ELFT>>(S);
    return make<ThumbV7ABSLongThunk<ELFT>>(S);
  }
  fatal("unrecognized relocation type");
}

template <class ELFT> static Thunk *addThunkMips(SymbolBody &S) {
  return make<MipsThunk<ELFT>>(S);
}

template <class ELFT> Thunk *addThunk(uint32_t RelocType, SymbolBody &S) {
  if (Config->EMachine == EM_ARM)
    return addThunkArm<ELFT>(RelocType, S);
  else if (Config->EMachine == EM_MIPS)
    return addThunkMips<ELFT>(S);
  llvm_unreachable("add Thunk only supported for ARM and Mips");
  return nullptr;
}

template Thunk *addThunk<ELF32LE>(uint32_t, SymbolBody &);
template Thunk *addThunk<ELF32BE>(uint32_t, SymbolBody &);
template Thunk *addThunk<ELF64LE>(uint32_t, SymbolBody &);
template Thunk *addThunk<ELF64BE>(uint32_t, SymbolBody &);
} // end namespace elf
} // end namespace lld
