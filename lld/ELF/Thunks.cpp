//===- Thunks.cpp --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This file contains both the Target independent and Target specific Thunk
// classes
//
// A Thunk Object represents a single Thunk that will be written to an
// InputSection when the InputSection contents are written. The InputSection
// maintains a list of Thunks that it owns.
//===---------------------------------------------------------------------===//

#include "Thunks.h"
#include "Error.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "OutputSections.h"
#include "Symbols.h"
#include "Target.h"
#include "llvm/Support/Allocator.h"

#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;

namespace lld {
namespace elf {

template <class ELFT> Thunk<ELFT>::~Thunk() {}

template <class ELFT>
Thunk<ELFT>::Thunk(const SymbolBody &D, const InputSection<ELFT> &O)
    : Destination(D), Owner(O), Offset(O.getThunkOff() + O.getThunksSize()) {}

template <class ELFT> typename ELFT::uint Thunk<ELFT>::getVA() const {
  return Owner.OutSec->getVA() + Owner.OutSecOff + Offset;
}

// ARM Target Thunks
template <class ELFT> static uint64_t getARMThunkDestVA(const SymbolBody &S) {
  return S.isInPlt() ? S.getPltVA<ELFT>() : S.getVA<ELFT>();
}

// Specific ARM Thunk implementations. The naming convention is:
// Source State, TargetState, Target Requirement, ABS or PI, Range
namespace {
template <class ELFT>
class ARMToThumbV7ABSLongThunk final : public Thunk<ELFT> {
public:
  uint32_t size() const override { return 12; }

  void writeTo(uint8_t *Buf) const override {
    const uint8_t ATData[] = {
        0x00, 0xc0, 0x00, 0xe3, // movw         ip,:lower16:S
        0x00, 0xc0, 0x40, 0xe3, // movt         ip,:upper16:S
        0x1c, 0xff, 0x2f, 0xe1, // bx   ip
    };
    uint64_t S = getARMThunkDestVA<ELFT>(this->Destination);
    memcpy(Buf, ATData, sizeof(ATData));
    Target->relocateOne(Buf, R_ARM_MOVW_ABS_NC, S);
    Target->relocateOne(Buf + 4, R_ARM_MOVT_ABS, S);
  }

  ARMToThumbV7ABSLongThunk(const SymbolBody &Destination,
                           const InputSection<ELFT> &Owner)
      : Thunk<ELFT>(Destination, Owner) {}
};

template <class ELFT> class ARMToThumbV7PILongThunk final : public Thunk<ELFT> {
public:
  uint32_t size() const override { return 16; }

  void writeTo(uint8_t *Buf) const override {
    const uint8_t ATData[] = {
        0xf0, 0xcf, 0x0f, 0xe3, // P:  movw ip,:lower16:S - (P + (L1-P) +8)
        0x00, 0xc0, 0x40, 0xe3, //     movt ip,:upper16:S - (P + (L1-P+4) +8)
        0x0f, 0xc0, 0x8c, 0xe0, // L1: add ip, ip, pc
        0x1c, 0xff, 0x2f, 0xe1, //     bx r12
    };
    uint64_t S = getARMThunkDestVA<ELFT>(this->Destination);
    uint64_t P = this->getVA();
    memcpy(Buf, ATData, sizeof(ATData));
    Target->relocateOne(Buf, R_ARM_MOVW_PREL_NC, S - P - 16);
    Target->relocateOne(Buf + 4, R_ARM_MOVT_PREL, S - P - 12);
  }

  ARMToThumbV7PILongThunk(const SymbolBody &Destination,
                          const InputSection<ELFT> &Owner)
      : Thunk<ELFT>(Destination, Owner) {}
};

template <class ELFT>
class ThumbToARMV7ABSLongThunk final : public Thunk<ELFT> {
public:
  uint32_t size() const override { return 10; }

  void writeTo(uint8_t *Buf) const override {
    const uint8_t TAData[] = {
        0x40, 0xf2, 0x00, 0x0c, // movw         ip, :lower16:S
        0xc0, 0xf2, 0x00, 0x0c, // movt         ip, :upper16:S
        0x60, 0x47,             // bx   ip
    };
    uint64_t S = getARMThunkDestVA<ELFT>(this->Destination);
    memcpy(Buf, TAData, sizeof(TAData));
    Target->relocateOne(Buf, R_ARM_THM_MOVW_ABS_NC, S);
    Target->relocateOne(Buf + 4, R_ARM_THM_MOVT_ABS, S);
  }

  ThumbToARMV7ABSLongThunk(const SymbolBody &Destination,
                           const InputSection<ELFT> &Owner)
      : Thunk<ELFT>(Destination, Owner) {}
};

template <class ELFT> class ThumbToARMV7PILongThunk final : public Thunk<ELFT> {
public:
  uint32_t size() const override { return 12; }

  void writeTo(uint8_t *Buf) const override {
    const uint8_t TAData[] = {
        0x4f, 0xf6, 0xf4, 0x7c, // P:  movw ip,:lower16:S - (P + (L1-P) + 4)
        0xc0, 0xf2, 0x00, 0x0c, //     movt ip,:upper16:S - (P + (L1-P+4) + 4)
        0xfc, 0x44,             // L1: add  r12, pc
        0x60, 0x47,             //     bx   r12
    };
    uint64_t S = getARMThunkDestVA<ELFT>(this->Destination);
    uint64_t P = this->getVA();
    memcpy(Buf, TAData, sizeof(TAData));
    Target->relocateOne(Buf, R_ARM_THM_MOVW_PREL_NC, S - P - 12);
    Target->relocateOne(Buf + 4, R_ARM_THM_MOVT_PREL, S - P - 8);
  }

  ThumbToARMV7PILongThunk(const SymbolBody &Destination,
                          const InputSection<ELFT> &Owner)
      : Thunk<ELFT>(Destination, Owner) {}
};

// Mips Thunks
// Only the MIPS LA25 Thunk is supported, the implementation is delegated
// to the MipsTargetInfo class in Target.cpp
template <class ELFT> class MipsThunk : public Thunk<ELFT> {
public:
  MipsThunk(const SymbolBody &Destination, const InputSection<ELFT> &Owner);
  uint32_t size() const override;
  void writeTo(uint8_t *Buf) const override;
};

template <class ELFT>
MipsThunk<ELFT>::MipsThunk(const SymbolBody &Destination,
                           const InputSection<ELFT> &Owner)
    : Thunk<ELFT>(Destination, Owner) {}

template <class ELFT> uint32_t MipsThunk<ELFT>::size() const { return 16; }

template <class ELFT> void MipsThunk<ELFT>::writeTo(uint8_t *Buf) const {
  const SymbolBody &D = this->Destination;
  uint64_t S = D.getVA<ELFT>();
  Target->writeThunk(Buf, S);
}
}

template <class ELFT>
static void addThunkARM(uint32_t RelocType, SymbolBody &S,
                        InputSection<ELFT> &IS) {
  if (S.hasThunk<ELFT>())
    // only one Thunk supported per symbol
    return;

  bool NeedsPI = Config->Pic || Config->Pie || Config->Shared;
  Thunk<ELFT> *T;
  BumpPtrAllocator &Alloc = IS.getFile()->Alloc;

  // ARM relocations need ARM to Thumb interworking Thunks, Thumb relocations
  // need Thumb to ARM relocations. Use position independent Thunks if we
  // require position independent code.
  switch (RelocType) {
  case R_ARM_PC24:
  case R_ARM_PLT32:
  case R_ARM_JUMP24:
    if (NeedsPI)
      T = new (Alloc) ARMToThumbV7PILongThunk<ELFT>(S, IS);
    else
      T = new (Alloc) ARMToThumbV7ABSLongThunk<ELFT>(S, IS);
    break;
  case R_ARM_THM_JUMP19:
  case R_ARM_THM_JUMP24:
    if (NeedsPI)
      T = new (Alloc) ThumbToARMV7PILongThunk<ELFT>(S, IS);
    else
      T = new (Alloc) ThumbToARMV7ABSLongThunk<ELFT>(S, IS);
    break;
  default:
    fatal("Unrecognised Relocation type\n");
  }

  // ARM Thunks are added to the same InputSection as the relocation. This
  // isn't strictly necessary but it makes it more likely that a limited range
  // branch can reach the Thunk, and it makes Thunks to the PLT section easier
  IS.addThunk(T);
  if (DefinedRegular<ELFT> *DR = dyn_cast<DefinedRegular<ELFT>>(&S))
    DR->ThunkData = T;
  else if (SharedSymbol<ELFT> *SH = dyn_cast<SharedSymbol<ELFT>>(&S))
    SH->ThunkData = T;
  else
    fatal("symbol not DefinedRegular or Shared\n");
}

template <class ELFT>
static void addThunkMips(uint32_t RelocType, SymbolBody &S,
                         InputSection<ELFT> &IS) {
  if (S.hasThunk<ELFT>())
    // only one Thunk supported per symbol
    return;

  // Mips Thunks are added to the InputSection defining S.
  auto *R = cast<DefinedRegular<ELFT>>(&S);
  auto *Sec = cast<InputSection<ELFT>>(R->Section);
  auto *T = new (IS.getFile()->Alloc) MipsThunk<ELFT>(S, *Sec);
  Sec->addThunk(T);
  R->ThunkData = T;
}

template <class ELFT>
void addThunk(uint32_t RelocType, SymbolBody &S, InputSection<ELFT> &IS) {
  if (Config->EMachine == EM_ARM)
    addThunkARM<ELFT>(RelocType, S, IS);
  else if (Config->EMachine == EM_MIPS)
    addThunkMips<ELFT>(RelocType, S, IS);
  else
    llvm_unreachable("add Thunk only supported for ARM and Mips");
}

template void addThunk<ELF32LE>(uint32_t, SymbolBody &,
                                InputSection<ELF32LE> &);
template void addThunk<ELF32BE>(uint32_t, SymbolBody &,
                                InputSection<ELF32BE> &);
template void addThunk<ELF64LE>(uint32_t, SymbolBody &,
                                InputSection<ELF64LE> &);
template void addThunk<ELF64BE>(uint32_t, SymbolBody &,
                                InputSection<ELF64BE> &);

template uint32_t Thunk<ELF32LE>::getVA() const;
template uint32_t Thunk<ELF32BE>::getVA() const;
template uint64_t Thunk<ELF64LE>::getVA() const;
template uint64_t Thunk<ELF64BE>::getVA() const;

} // namespace elf
} // namespace lld
