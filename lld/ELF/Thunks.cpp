//===- Thunks.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "InputFiles.h"
#include "InputSection.h"
#include "OutputSections.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Common/CommonLinkerContext.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <cstring>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace {

// Base class for AArch64 thunks.
//
// An AArch64 thunk may be either short or long. A short thunk is simply a
// branch (B) instruction, and it may be used to call AArch64 functions when the
// distance from the thunk to the target is less than 128MB. Long thunks can
// branch to any virtual address and they are implemented in the derived
// classes. This class tries to create a short thunk if the target is in range,
// otherwise it creates a long thunk. When BTI is enabled indirect branches
// must land on a BTI instruction. If the destination does not have a BTI
// instruction mayNeedLandingPad is set to true and Thunk::landingPad points
// to an alternative entry point with a BTI.
class AArch64Thunk : public Thunk {
public:
  AArch64Thunk(Ctx &ctx, Symbol &dest, int64_t addend, bool mayNeedLandingPad)
      : Thunk(ctx, dest, addend), mayNeedLandingPad(mayNeedLandingPad) {}
  bool getMayUseShortThunk();
  void writeTo(uint8_t *buf) override;
  bool needsSyntheticLandingPad() override;

protected:
  bool mayNeedLandingPad;

private:
  bool mayUseShortThunk = true;
  virtual void writeLong(uint8_t *buf) = 0;
  // A thunk may be written out as a short or long, and we may not know which
  // type at thunk creation time. In some thunk implementations the long thunk
  // has additional mapping symbols. Thus function can be overridden to add
  // these additional mapping symbols.
  virtual void addLongMapSyms() {}
};

// AArch64 long range Thunks.
class AArch64ABSLongThunk final : public AArch64Thunk {
public:
  AArch64ABSLongThunk(Ctx &ctx, Symbol &dest, int64_t addend,
                      bool mayNeedLandingPad)
      : AArch64Thunk(ctx, dest, addend, mayNeedLandingPad) {}
  uint32_t size() override { return getMayUseShortThunk() ? 4 : 16; }
  void addSymbols(ThunkSection &isec) override;

private:
  void writeLong(uint8_t *buf) override;
  void addLongMapSyms() override;
  ThunkSection *tsec = nullptr;
};

// AArch64 long range Thunks compatible with execute-only code.
class AArch64ABSXOLongThunk final : public AArch64Thunk {
public:
  AArch64ABSXOLongThunk(Ctx &ctx, Symbol &dest, int64_t addend,
                        bool mayNeedLandingPad)
      : AArch64Thunk(ctx, dest, addend, mayNeedLandingPad) {}
  uint32_t size() override { return getMayUseShortThunk() ? 4 : 20; }
  void addSymbols(ThunkSection &sec) override;

private:
  void writeLong(uint8_t *buf) override;
};

class AArch64ADRPThunk final : public AArch64Thunk {
public:
  AArch64ADRPThunk(Ctx &ctx, Symbol &dest, int64_t addend,
                   bool mayNeedLandingPad)
      : AArch64Thunk(ctx, dest, addend, mayNeedLandingPad) {}
  uint32_t size() override { return getMayUseShortThunk() ? 4 : 12; }
  void addSymbols(ThunkSection &isec) override;

private:
  void writeLong(uint8_t *buf) override;
};

// AArch64 BTI Landing Pad
// When BTI is enabled indirect branches must land on a BTI
// compatible instruction. When the destination does not have a
// BTI compatible instruction a Thunk doing an indirect branch
// targets a Landing Pad Thunk that direct branches to the target.
class AArch64BTILandingPadThunk final : public Thunk {
public:
  AArch64BTILandingPadThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : Thunk(ctx, dest, addend) {}

  uint32_t size() override { return getMayUseShortThunk() ? 4 : 8; }
  void addSymbols(ThunkSection &isec) override;
  void writeTo(uint8_t *buf) override;

private:
  bool getMayUseShortThunk();
  void writeLong(uint8_t *buf);
  bool mayUseShortThunk = true;
};

// Base class for ARM thunks.
//
// An ARM thunk may be either short or long. A short thunk is simply a branch
// (B) instruction, and it may be used to call ARM functions when the distance
// from the thunk to the target is less than 32MB. Long thunks can branch to any
// virtual address and can switch between ARM and Thumb, and they are
// implemented in the derived classes. This class tries to create a short thunk
// if the target is in range, otherwise it creates a long thunk.
class ARMThunk : public Thunk {
public:
  ARMThunk(Ctx &ctx, Symbol &dest, int64_t addend) : Thunk(ctx, dest, addend) {}

  bool getMayUseShortThunk();
  uint32_t size() override { return getMayUseShortThunk() ? 4 : sizeLong(); }
  void writeTo(uint8_t *buf) override;
  bool isCompatibleWith(const InputSection &isec,
                        const Relocation &rel) const override;

  // Returns the size of a long thunk.
  virtual uint32_t sizeLong() = 0;

  // Writes a long thunk to Buf.
  virtual void writeLong(uint8_t *buf) = 0;

private:
  // This field tracks whether all previously considered layouts would allow
  // this thunk to be short. If we have ever needed a long thunk, we always
  // create a long thunk, even if the thunk may be short given the current
  // distance to the target. We do this because transitioning from long to short
  // can create layout oscillations in certain corner cases which would prevent
  // the layout from converging.
  bool mayUseShortThunk = true;
  // See comment in AArch64Thunk.
  virtual void addLongMapSyms() {}
};

// Base class for Thumb-2 thunks.
//
// This class is similar to ARMThunk, but it uses the Thumb-2 B.W instruction
// which has a range of 16MB.
class ThumbThunk : public Thunk {
public:
  ThumbThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : Thunk(ctx, dest, addend) {
    alignment = 2;
  }

  bool getMayUseShortThunk();
  uint32_t size() override { return getMayUseShortThunk() ? 4 : sizeLong(); }
  void writeTo(uint8_t *buf) override;
  bool isCompatibleWith(const InputSection &isec,
                        const Relocation &rel) const override;

  // Returns the size of a long thunk.
  virtual uint32_t sizeLong() = 0;

  // Writes a long thunk to Buf.
  virtual void writeLong(uint8_t *buf) = 0;

private:
  // See comment in ARMThunk above.
  bool mayUseShortThunk = true;
  // See comment in AArch64Thunk.
  virtual void addLongMapSyms() {}
};

// Specific ARM Thunk implementations. The naming convention is:
// Source State, TargetState, Target Requirement, ABS or PI, Range
class ARMV7ABSLongThunk final : public ARMThunk {
public:
  ARMV7ABSLongThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : ARMThunk(ctx, dest, addend) {}

  uint32_t sizeLong() override { return 12; }
  void writeLong(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
};

class ARMV7PILongThunk final : public ARMThunk {
public:
  ARMV7PILongThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : ARMThunk(ctx, dest, addend) {}

  uint32_t sizeLong() override { return 16; }
  void writeLong(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
};

class ThumbV7ABSLongThunk final : public ThumbThunk {
public:
  ThumbV7ABSLongThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : ThumbThunk(ctx, dest, addend) {}

  uint32_t sizeLong() override { return 10; }
  void writeLong(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
};

class ThumbV7PILongThunk final : public ThumbThunk {
public:
  ThumbV7PILongThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : ThumbThunk(ctx, dest, addend) {}

  uint32_t sizeLong() override { return 12; }
  void writeLong(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
};

// Implementations of Thunks for Arm v6-M. Only Thumb instructions are permitted
class ThumbV6MABSLongThunk final : public ThumbThunk {
public:
  ThumbV6MABSLongThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : ThumbThunk(ctx, dest, addend) {}

  uint32_t sizeLong() override { return 12; }
  void writeLong(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;

private:
  void addLongMapSyms() override;
  ThunkSection *tsec = nullptr;
};

class ThumbV6MABSXOLongThunk final : public ThumbThunk {
public:
  ThumbV6MABSXOLongThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : ThumbThunk(ctx, dest, addend) {}

  uint32_t sizeLong() override { return 20; }
  void writeLong(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
};

class ThumbV6MPILongThunk final : public ThumbThunk {
public:
  ThumbV6MPILongThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : ThumbThunk(ctx, dest, addend) {}

  uint32_t sizeLong() override { return 16; }
  void writeLong(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;

private:
  void addLongMapSyms() override;
  ThunkSection *tsec = nullptr;
};

// Architectures v4, v5 and v6 do not support the movt/movw instructions. v5 and
// v6 support BLX to which BL instructions can be rewritten inline. There are no
// Thumb entrypoints for v5 and v6 as there is no Thumb branch instruction on
// these architecture that can result in a thunk.

// LDR on v5 and v6 can switch processor state, so for v5 and v6,
// ARMV5LongLdrPcThunk can be used for both Arm->Arm and Arm->Thumb calls. v4
// can also use this thunk, but only for Arm->Arm calls.
class ARMV5LongLdrPcThunk final : public ARMThunk {
public:
  ARMV5LongLdrPcThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : ARMThunk(ctx, dest, addend) {}

  uint32_t sizeLong() override { return 8; }
  void writeLong(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;

private:
  void addLongMapSyms() override;
  ThunkSection *tsec = nullptr;
};

// Implementations of Thunks for v4. BLX is not supported, and loads
// will not invoke Arm/Thumb state changes.
class ARMV4PILongBXThunk final : public ARMThunk {
public:
  ARMV4PILongBXThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : ARMThunk(ctx, dest, addend) {}

  uint32_t sizeLong() override { return 16; }
  void writeLong(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;

private:
  void addLongMapSyms() override;
  ThunkSection *tsec = nullptr;
};

class ARMV4PILongThunk final : public ARMThunk {
public:
  ARMV4PILongThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : ARMThunk(ctx, dest, addend) {}

  uint32_t sizeLong() override { return 12; }
  void writeLong(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;

private:
  void addLongMapSyms() override;
  ThunkSection *tsec = nullptr;
};

class ThumbV4PILongBXThunk final : public ThumbThunk {
public:
  ThumbV4PILongBXThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : ThumbThunk(ctx, dest, addend) {}

  uint32_t sizeLong() override { return 16; }
  void writeLong(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;

private:
  void addLongMapSyms() override;
  ThunkSection *tsec = nullptr;
};

class ThumbV4PILongThunk final : public ThumbThunk {
public:
  ThumbV4PILongThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : ThumbThunk(ctx, dest, addend) {}

  uint32_t sizeLong() override { return 20; }
  void writeLong(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;

private:
  void addLongMapSyms() override;
  ThunkSection *tsec = nullptr;
};

class ARMV4ABSLongBXThunk final : public ARMThunk {
public:
  ARMV4ABSLongBXThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : ARMThunk(ctx, dest, addend) {}

  uint32_t sizeLong() override { return 12; }
  void writeLong(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;

private:
  void addLongMapSyms() override;
  ThunkSection *tsec = nullptr;
};

class ThumbV4ABSLongBXThunk final : public ThumbThunk {
public:
  ThumbV4ABSLongBXThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : ThumbThunk(ctx, dest, addend) {}

  uint32_t sizeLong() override { return 12; }
  void writeLong(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;

private:
  void addLongMapSyms() override;
  ThunkSection *tsec = nullptr;
};

class ThumbV4ABSLongThunk final : public ThumbThunk {
public:
  ThumbV4ABSLongThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : ThumbThunk(ctx, dest, addend) {}

  uint32_t sizeLong() override { return 16; }
  void writeLong(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;

private:
  void addLongMapSyms() override;
  ThunkSection *tsec = nullptr;
};

// The AVR devices need thunks for R_AVR_LO8_LDI_GS/R_AVR_HI8_LDI_GS
// when their destination is out of range [0, 0x1ffff].
class AVRThunk : public Thunk {
public:
  AVRThunk(Ctx &ctx, Symbol &dest, int64_t addend) : Thunk(ctx, dest, addend) {}
  uint32_t size() override { return 4; }
  void writeTo(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
};

// Hexagon CPUs need thunks for R_HEX_B{9,1{3,5},22}_PCREL,
// R_HEX_{,GD_}PLT_B22_PCREL when their destination is out of
// range.
class HexagonThunk : public Thunk {
public:
  HexagonThunk(Ctx &ctx, const InputSection &isec, Relocation &rel,
               Symbol &dest)
      : Thunk(ctx, dest, 0), relOffset(rel.offset) {
    alignment = 4;
  }
  uint32_t relOffset;
  uint32_t size() override { return ctx.arg.isPic ? 12 : 8; }
  void writeTo(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
};

// MIPS LA25 thunk
class MipsThunk final : public Thunk {
public:
  MipsThunk(Ctx &ctx, Symbol &dest) : Thunk(ctx, dest, 0) {}

  uint32_t size() override { return 16; }
  void writeTo(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
  InputSection *getTargetInputSection() const override;
};

// microMIPS R2-R5 LA25 thunk
class MicroMipsThunk final : public Thunk {
public:
  MicroMipsThunk(Ctx &ctx, Symbol &dest) : Thunk(ctx, dest, 0) {}

  uint32_t size() override { return 14; }
  void writeTo(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
  InputSection *getTargetInputSection() const override;
};

// microMIPS R6 LA25 thunk
class MicroMipsR6Thunk final : public Thunk {
public:
  MicroMipsR6Thunk(Ctx &ctx, Symbol &dest) : Thunk(ctx, dest, 0) {}

  uint32_t size() override { return 12; }
  void writeTo(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
  InputSection *getTargetInputSection() const override;
};

class PPC32PltCallStub final : public Thunk {
public:
  // For R_PPC_PLTREL24, Thunk::addend records the addend which will be used to
  // decide the offsets in the call stub.
  PPC32PltCallStub(Ctx &ctx, const InputSection &isec, const Relocation &rel,
                   Symbol &dest)
      : Thunk(ctx, dest, rel.addend), file(isec.file) {}
  uint32_t size() override { return 16; }
  void writeTo(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
  bool isCompatibleWith(const InputSection &isec, const Relocation &rel) const override;

private:
  // Records the call site of the call stub.
  const InputFile *file;
};

class PPC32LongThunk final : public Thunk {
public:
  PPC32LongThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : Thunk(ctx, dest, addend) {}
  uint32_t size() override { return ctx.arg.isPic ? 32 : 16; }
  void writeTo(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
};

// PPC64 Plt call stubs.
// Any call site that needs to call through a plt entry needs a call stub in
// the .text section. The call stub is responsible for:
// 1) Saving the toc-pointer to the stack.
// 2) Loading the target functions address from the procedure linkage table into
//    r12 for use by the target functions global entry point, and into the count
//    register.
// 3) Transferring control to the target function through an indirect branch.
class PPC64PltCallStub final : public Thunk {
public:
  PPC64PltCallStub(Ctx &ctx, Symbol &dest) : Thunk(ctx, dest, 0) {}
  uint32_t size() override { return 20; }
  void writeTo(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
  bool isCompatibleWith(const InputSection &isec,
                        const Relocation &rel) const override;
};

// PPC64 R2 Save Stub
// When the caller requires a valid R2 TOC pointer but the callee does not
// require a TOC pointer and the callee cannot guarantee that it doesn't
// clobber R2 then we need to save R2. This stub:
// 1) Saves the TOC pointer to the stack.
// 2) Tail calls the callee.
class PPC64R2SaveStub final : public Thunk {
public:
  PPC64R2SaveStub(Ctx &ctx, Symbol &dest, int64_t addend)
      : Thunk(ctx, dest, addend) {
    alignment = 16;
  }

  // To prevent oscillations in layout when moving from short to long thunks
  // we make sure that once a thunk has been set to long it cannot go back.
  bool getMayUseShortThunk() {
    if (!mayUseShortThunk)
      return false;
    if (!isInt<26>(computeOffset())) {
      mayUseShortThunk = false;
      return false;
    }
    return true;
  }
  uint32_t size() override { return getMayUseShortThunk() ? 8 : 32; }
  void writeTo(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
  bool isCompatibleWith(const InputSection &isec,
                        const Relocation &rel) const override;

private:
  // Transitioning from long to short can create layout oscillations in
  // certain corner cases which would prevent the layout from converging.
  // This is similar to the handling for ARMThunk.
  bool mayUseShortThunk = true;
  int64_t computeOffset() const {
    return destination.getVA(ctx) - (getThunkTargetSym()->getVA(ctx) + 4);
  }
};

// PPC64 R12 Setup Stub
// When a caller that does not maintain TOC calls a target which may possibly
// use TOC (either non-preemptible with localentry>1 or preemptible), we need to
// set r12 to satisfy the requirement of the global entry point.
class PPC64R12SetupStub final : public Thunk {
public:
  PPC64R12SetupStub(Ctx &ctx, Symbol &dest, bool gotPlt)
      : Thunk(ctx, dest, 0), gotPlt(gotPlt) {
    alignment = 16;
  }
  uint32_t size() override { return 32; }
  void writeTo(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
  bool isCompatibleWith(const InputSection &isec,
                        const Relocation &rel) const override;

private:
  bool gotPlt;
};

// A bl instruction uses a signed 24 bit offset, with an implicit 4 byte
// alignment. This gives a possible 26 bits of 'reach'. If the call offset is
// larger than that we need to emit a long-branch thunk. The target address
// of the callee is stored in a table to be accessed TOC-relative. Since the
// call must be local (a non-local call will have a PltCallStub instead) the
// table stores the address of the callee's local entry point. For
// position-independent code a corresponding relative dynamic relocation is
// used.
class PPC64LongBranchThunk : public Thunk {
public:
  uint32_t size() override { return 32; }
  void writeTo(uint8_t *buf) override;
  void addSymbols(ThunkSection &isec) override;
  bool isCompatibleWith(const InputSection &isec,
                        const Relocation &rel) const override;

protected:
  PPC64LongBranchThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : Thunk(ctx, dest, addend) {}
};

class PPC64PILongBranchThunk final : public PPC64LongBranchThunk {
public:
  PPC64PILongBranchThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : PPC64LongBranchThunk(ctx, dest, addend) {
    assert(!dest.isPreemptible);
    if (std::optional<uint32_t> index =
            ctx.in.ppc64LongBranchTarget->addEntry(&dest, addend)) {
      ctx.mainPart->relaDyn->addRelativeReloc(
          ctx.target->relativeRel, *ctx.in.ppc64LongBranchTarget,
          *index * UINT64_C(8), dest,
          addend + getPPC64GlobalEntryToLocalEntryOffset(ctx, dest.stOther),
          ctx.target->symbolicRel, R_ABS);
    }
  }
};

class PPC64PDLongBranchThunk final : public PPC64LongBranchThunk {
public:
  PPC64PDLongBranchThunk(Ctx &ctx, Symbol &dest, int64_t addend)
      : PPC64LongBranchThunk(ctx, dest, addend) {
    ctx.in.ppc64LongBranchTarget->addEntry(&dest, addend);
  }
};

} // end anonymous namespace

Defined *Thunk::addSymbol(StringRef name, uint8_t type, uint64_t value,
                          InputSectionBase &section) {
  Defined *d =
      addSyntheticLocal(ctx, name, type, value + offset, /*size=*/0, section);
  syms.push_back(d);
  return d;
}

void Thunk::setOffset(uint64_t newOffset) {
  for (Defined *d : syms)
    d->value = d->value - offset + newOffset;
  offset = newOffset;
}

// AArch64 Thunk base class.
static uint64_t getAArch64ThunkDestVA(Ctx &ctx, const Symbol &s, int64_t a) {
  uint64_t v = s.isInPlt(ctx) ? s.getPltVA(ctx) : s.getVA(ctx, a);
  return v;
}

bool AArch64Thunk::getMayUseShortThunk() {
  if (!mayUseShortThunk)
    return false;
  uint64_t s = getAArch64ThunkDestVA(ctx, destination, addend);
  uint64_t p = getThunkTargetSym()->getVA(ctx);
  mayUseShortThunk = llvm::isInt<28>(s - p);
  if (!mayUseShortThunk)
    addLongMapSyms();
  return mayUseShortThunk;
}

void AArch64Thunk::writeTo(uint8_t *buf) {
  if (!getMayUseShortThunk()) {
    writeLong(buf);
    return;
  }
  uint64_t s = getAArch64ThunkDestVA(ctx, destination, addend);
  uint64_t p = getThunkTargetSym()->getVA(ctx);
  write32(ctx, buf, 0x14000000); // b S
  ctx.target->relocateNoSym(buf, R_AARCH64_CALL26, s - p);
}

bool AArch64Thunk::needsSyntheticLandingPad() {
  // Short Thunks use a direct branch, no synthetic landing pad
  // required.
  return mayNeedLandingPad && !getMayUseShortThunk();
}

// AArch64 long range Thunks.
void AArch64ABSLongThunk::writeLong(uint8_t *buf) {
  const uint8_t data[] = {
    0x50, 0x00, 0x00, 0x58, //     ldr x16, L0
    0x00, 0x02, 0x1f, 0xd6, //     br  x16
    0x00, 0x00, 0x00, 0x00, // L0: .xword S
    0x00, 0x00, 0x00, 0x00,
  };
  // If mayNeedLandingPad is true then destination is an
  // AArch64BTILandingPadThunk that defines landingPad.
  assert(!mayNeedLandingPad || landingPad != nullptr);
  uint64_t s = mayNeedLandingPad
                   ? landingPad->getVA(ctx, 0)
                   : getAArch64ThunkDestVA(ctx, destination, addend);
  memcpy(buf, data, sizeof(data));
  ctx.target->relocateNoSym(buf + 8, R_AARCH64_ABS64, s);
}

void AArch64ABSLongThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__AArch64AbsLongThunk_" + destination.getName()),
            STT_FUNC, 0, isec);
  addSymbol("$x", STT_NOTYPE, 0, isec);
  tsec = &isec;
  (void)getMayUseShortThunk();
}

void AArch64ABSLongThunk::addLongMapSyms() {
  addSymbol("$d", STT_NOTYPE, 8, *tsec);
  // The ldr in the long Thunk requires 8-byte alignment when
  // unaligned accesses are disabled.
  alignment = 8;
}

void AArch64ABSXOLongThunk::writeLong(uint8_t *buf) {
  const uint8_t data[] = {
      0x10, 0x00, 0x80, 0xd2, // movz x16, :abs_g0_nc:S, lsl #0
      0x10, 0x00, 0xa0, 0xf2, // movk x16, :abs_g1_nc:S, lsl #16
      0x10, 0x00, 0xc0, 0xf2, // movk x16, :abs_g2_nc:S, lsl #32
      0x10, 0x00, 0xe0, 0xf2, // movk x16, :abs_g3:S,    lsl #48
      0x00, 0x02, 0x1f, 0xd6, // br   x16
  };
  // If mayNeedLandingPad is true then destination is an
  // AArch64BTILandingPadThunk that defines landingPad.
  assert(!mayNeedLandingPad || landingPad != nullptr);
  uint64_t s = mayNeedLandingPad
                   ? landingPad->getVA(ctx, 0)
                   : getAArch64ThunkDestVA(ctx, destination, addend);
  memcpy(buf, data, sizeof(data));
  ctx.target->relocateNoSym(buf + 0, R_AARCH64_MOVW_UABS_G0_NC, s);
  ctx.target->relocateNoSym(buf + 4, R_AARCH64_MOVW_UABS_G1_NC, s);
  ctx.target->relocateNoSym(buf + 8, R_AARCH64_MOVW_UABS_G2_NC, s);
  ctx.target->relocateNoSym(buf + 12, R_AARCH64_MOVW_UABS_G3, s);
}

void AArch64ABSXOLongThunk::addSymbols(ThunkSection &sec) {
  addSymbol(ctx.saver.save("__AArch64AbsXOLongThunk_" + destination.getName()),
            STT_FUNC, 0, sec);
  addSymbol("$x", STT_NOTYPE, 0, sec);
}

// This Thunk has a maximum range of 4Gb, this is sufficient for all programs
// using the small code model, including pc-relative ones. At time of writing
// clang and gcc do not support the large code model for position independent
// code so it is safe to use this for position independent thunks without
// worrying about the destination being more than 4Gb away.
void AArch64ADRPThunk::writeLong(uint8_t *buf) {
  const uint8_t data[] = {
      0x10, 0x00, 0x00, 0x90, // adrp x16, Dest R_AARCH64_ADR_PREL_PG_HI21(Dest)
      0x10, 0x02, 0x00, 0x91, // add  x16, x16, R_AARCH64_ADD_ABS_LO12_NC(Dest)
      0x00, 0x02, 0x1f, 0xd6, // br   x16
  };
  // if mayNeedLandingPad is true then destination is an
  // AArch64BTILandingPadThunk that defines landingPad.
  assert(!mayNeedLandingPad || landingPad != nullptr);
  uint64_t s = mayNeedLandingPad
                   ? landingPad->getVA(ctx, 0)
                   : getAArch64ThunkDestVA(ctx, destination, addend);
  uint64_t p = getThunkTargetSym()->getVA(ctx);
  memcpy(buf, data, sizeof(data));
  ctx.target->relocateNoSym(buf, R_AARCH64_ADR_PREL_PG_HI21,
                            getAArch64Page(s) - getAArch64Page(p));
  ctx.target->relocateNoSym(buf + 4, R_AARCH64_ADD_ABS_LO12_NC, s);
}

void AArch64ADRPThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__AArch64ADRPThunk_" + destination.getName()),
            STT_FUNC, 0, isec);
  addSymbol("$x", STT_NOTYPE, 0, isec);
}

void AArch64BTILandingPadThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__AArch64BTIThunk_" + destination.getName()),
            STT_FUNC, 0, isec);
  addSymbol("$x", STT_NOTYPE, 0, isec);
}

void AArch64BTILandingPadThunk::writeTo(uint8_t *buf) {
  if (!getMayUseShortThunk()) {
    writeLong(buf);
    return;
  }
  write32(ctx, buf, 0xd503245f); // BTI c
  // Control falls through to target in following section.
}

bool AArch64BTILandingPadThunk::getMayUseShortThunk() {
  if (!mayUseShortThunk)
    return false;
  // If the target is the following instruction then we can fall
  // through without the indirect branch.
  uint64_t s = destination.getVA(ctx, addend);
  uint64_t p = getThunkTargetSym()->getVA(ctx);
  // This function is called before addresses are stable.  We need to
  // work out the range from the thunk to the next section but the
  // address of the start of the next section depends on the size of
  // the thunks in the previous pass.  s - p + offset == 0 represents
  // the first pass where the Thunk and following section are assigned
  // the same offset.  s - p <= 4 is the last Thunk in the Thunk
  // Section.
  mayUseShortThunk = (s - p + offset == 0 || s - p <= 4);
  return mayUseShortThunk;
}

void AArch64BTILandingPadThunk::writeLong(uint8_t *buf) {
  uint64_t s = destination.getVA(ctx, addend);
  uint64_t p = getThunkTargetSym()->getVA(ctx) + 4;
  write32(ctx, buf, 0xd503245f);     // BTI c
  write32(ctx, buf + 4, 0x14000000); // B S
  ctx.target->relocateNoSym(buf + 4, R_AARCH64_CALL26, s - p);
}

// ARM Target Thunks
static uint64_t getARMThunkDestVA(Ctx &ctx, const Symbol &s) {
  uint64_t v = s.isInPlt(ctx) ? s.getPltVA(ctx) : s.getVA(ctx);
  return SignExtend64<32>(v);
}

// This function returns true if the target is not Thumb and is within 2^26, and
// it has not previously returned false (see comment for mayUseShortThunk).
bool ARMThunk::getMayUseShortThunk() {
  if (!mayUseShortThunk)
    return false;
  uint64_t s = getARMThunkDestVA(ctx, destination);
  if (s & 1) {
    mayUseShortThunk = false;
    addLongMapSyms();
    return false;
  }
  uint64_t p = getThunkTargetSym()->getVA(ctx);
  int64_t offset = s - p - 8;
  mayUseShortThunk = llvm::isInt<26>(offset);
  if (!mayUseShortThunk)
    addLongMapSyms();
  return mayUseShortThunk;
}

void ARMThunk::writeTo(uint8_t *buf) {
  if (!getMayUseShortThunk()) {
    writeLong(buf);
    return;
  }

  uint64_t s = getARMThunkDestVA(ctx, destination);
  uint64_t p = getThunkTargetSym()->getVA(ctx);
  int64_t offset = s - p - 8;
  write32(ctx, buf, 0xea000000); // b S
  ctx.target->relocateNoSym(buf, R_ARM_JUMP24, offset);
}

bool ARMThunk::isCompatibleWith(const InputSection &isec,
                                const Relocation &rel) const {
  // v4T does not have BLX, so also deny R_ARM_THM_CALL
  if (!ctx.arg.armHasBlx && rel.type == R_ARM_THM_CALL)
    return false;

  // Thumb branch relocations can't use BLX
  return rel.type != R_ARM_THM_JUMP19 && rel.type != R_ARM_THM_JUMP24;
}

// This function returns true if:
// the target is Thumb
// && is within branch range
// && this function has not previously returned false
//    (see comment for mayUseShortThunk)
// && the arch supports Thumb branch range extension.
bool ThumbThunk::getMayUseShortThunk() {
  if (!mayUseShortThunk)
    return false;
  uint64_t s = getARMThunkDestVA(ctx, destination);
  // To use a short thunk the destination must be Thumb and the target must
  // have the wide branch instruction B.w. This instruction is included when
  // Thumb 2 is present, or in v8-M (and above) baseline architectures.
  // armJ1J2BranchEncoding is available in all architectures with a profile and
  // the one v6 CPU that implements Thumb 2 (Arm1156t2-s).
  // Movt and Movw instructions require Thumb 2 or v8-M baseline.
  if ((s & 1) == 0 || !ctx.arg.armJ1J2BranchEncoding ||
      !ctx.arg.armHasMovtMovw) {
    mayUseShortThunk = false;
    addLongMapSyms();
    return false;
  }
  uint64_t p = getThunkTargetSym()->getVA(ctx) & ~1;
  int64_t offset = s - p - 4;
  mayUseShortThunk = llvm::isInt<25>(offset);
  if (!mayUseShortThunk)
    addLongMapSyms();
  return mayUseShortThunk;
}

void ThumbThunk::writeTo(uint8_t *buf) {
  if (!getMayUseShortThunk()) {
    writeLong(buf);
    return;
  }

  uint64_t s = getARMThunkDestVA(ctx, destination);
  uint64_t p = getThunkTargetSym()->getVA(ctx);
  int64_t offset = s - p - 4;
  write16(ctx, buf + 0, 0xf000); // b.w S
  write16(ctx, buf + 2, 0xb000);
  ctx.target->relocateNoSym(buf, R_ARM_THM_JUMP24, offset);
}

bool ThumbThunk::isCompatibleWith(const InputSection &isec,
                                  const Relocation &rel) const {
  // v4T does not have BLX, so also deny R_ARM_CALL
  if (!ctx.arg.armHasBlx && rel.type == R_ARM_CALL)
    return false;

  // ARM branch relocations can't use BLX
  return rel.type != R_ARM_JUMP24 && rel.type != R_ARM_PC24 && rel.type != R_ARM_PLT32;
}

void ARMV7ABSLongThunk::writeLong(uint8_t *buf) {
  write32(ctx, buf + 0, 0xe300c000); // movw ip,:lower16:S
  write32(ctx, buf + 4, 0xe340c000); // movt ip,:upper16:S
  write32(ctx, buf + 8, 0xe12fff1c); // bx   ip
  uint64_t s = getARMThunkDestVA(ctx, destination);
  ctx.target->relocateNoSym(buf, R_ARM_MOVW_ABS_NC, s);
  ctx.target->relocateNoSym(buf + 4, R_ARM_MOVT_ABS, s);
}

void ARMV7ABSLongThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__ARMv7ABSLongThunk_" + destination.getName()),
            STT_FUNC, 0, isec);
  addSymbol("$a", STT_NOTYPE, 0, isec);
}

void ThumbV7ABSLongThunk::writeLong(uint8_t *buf) {
  write16(ctx, buf + 0, 0xf240); // movw ip, :lower16:S
  write16(ctx, buf + 2, 0x0c00);
  write16(ctx, buf + 4, 0xf2c0); // movt ip, :upper16:S
  write16(ctx, buf + 6, 0x0c00);
  write16(ctx, buf + 8, 0x4760); // bx   ip
  uint64_t s = getARMThunkDestVA(ctx, destination);
  ctx.target->relocateNoSym(buf, R_ARM_THM_MOVW_ABS_NC, s);
  ctx.target->relocateNoSym(buf + 4, R_ARM_THM_MOVT_ABS, s);
}

void ThumbV7ABSLongThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__Thumbv7ABSLongThunk_" + destination.getName()),
            STT_FUNC, 1, isec);
  addSymbol("$t", STT_NOTYPE, 0, isec);
}

void ARMV7PILongThunk::writeLong(uint8_t *buf) {
  write32(ctx, buf + 0,
          0xe30fcff0); // P:  movw ip,:lower16:S - (P + (L1-P) + 8)
  write32(ctx, buf + 4,
          0xe340c000); //     movt ip,:upper16:S - (P + (L1-P) + 8)
  write32(ctx, buf + 8, 0xe08cc00f);  // L1: add  ip, ip, pc
  write32(ctx, buf + 12, 0xe12fff1c); //     bx   ip
  uint64_t s = getARMThunkDestVA(ctx, destination);
  uint64_t p = getThunkTargetSym()->getVA(ctx);
  int64_t offset = s - p - 16;
  ctx.target->relocateNoSym(buf, R_ARM_MOVW_PREL_NC, offset);
  ctx.target->relocateNoSym(buf + 4, R_ARM_MOVT_PREL, offset);
}

void ARMV7PILongThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__ARMV7PILongThunk_" + destination.getName()),
            STT_FUNC, 0, isec);
  addSymbol("$a", STT_NOTYPE, 0, isec);
}

void ThumbV7PILongThunk::writeLong(uint8_t *buf) {
  write16(ctx, buf + 0, 0xf64f); // P:  movw ip,:lower16:S - (P + (L1-P) + 4)
  write16(ctx, buf + 2, 0x7cf4);
  write16(ctx, buf + 4, 0xf2c0); //     movt ip,:upper16:S - (P + (L1-P) + 4)
  write16(ctx, buf + 6, 0x0c00);
  write16(ctx, buf + 8, 0x44fc);  // L1: add  ip, pc
  write16(ctx, buf + 10, 0x4760); //     bx   ip
  uint64_t s = getARMThunkDestVA(ctx, destination);
  uint64_t p = getThunkTargetSym()->getVA(ctx) & ~0x1;
  int64_t offset = s - p - 12;
  ctx.target->relocateNoSym(buf, R_ARM_THM_MOVW_PREL_NC, offset);
  ctx.target->relocateNoSym(buf + 4, R_ARM_THM_MOVT_PREL, offset);
}

void ThumbV7PILongThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__ThumbV7PILongThunk_" + destination.getName()),
            STT_FUNC, 1, isec);
  addSymbol("$t", STT_NOTYPE, 0, isec);
}

void ThumbV6MABSLongThunk::writeLong(uint8_t *buf) {
  // Most Thumb instructions cannot access the high registers r8 - r15. As the
  // only register we can corrupt is r12 we must instead spill a low register
  // to the stack to use as a scratch register. We push r1 even though we
  // don't need to get some space to use for the return address.
  write16(ctx, buf + 0, 0xb403); // push {r0, r1} ; Obtain scratch registers
  write16(ctx, buf + 2, 0x4801); // ldr r0, [pc, #4] ; L1
  write16(ctx, buf + 4, 0x9001); // str r0, [sp, #4] ; SP + 4 = S
  write16(ctx, buf + 6, 0xbd01); // pop {r0, pc} ; restore r0 and branch to dest
  write32(ctx, buf + 8, 0x00000000); // L1: .word S
  uint64_t s = getARMThunkDestVA(ctx, destination);
  ctx.target->relocateNoSym(buf + 8, R_ARM_ABS32, s);
}

void ThumbV6MABSLongThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__Thumbv6MABSLongThunk_" + destination.getName()),
            STT_FUNC, 1, isec);
  addSymbol("$t", STT_NOTYPE, 0, isec);
  tsec = &isec;
  (void)getMayUseShortThunk();
}

void ThumbV6MABSLongThunk::addLongMapSyms() {
  addSymbol("$d", STT_NOTYPE, 8, *tsec);
}

void ThumbV6MABSXOLongThunk::writeLong(uint8_t *buf) {
  // Most Thumb instructions cannot access the high registers r8 - r15. As the
  // only register we can corrupt is r12 we must instead spill a low register
  // to the stack to use as a scratch register. We push r1 even though we
  // don't need to get some space to use for the return address.
  write16(ctx, buf + 0, 0xb403);  // push {r0, r1} ; Obtain scratch registers
  write16(ctx, buf + 2, 0x2000);  // movs r0, :upper8_15:S
  write16(ctx, buf + 4, 0x0200);  // lsls r0, r0, #8
  write16(ctx, buf + 6, 0x3000);  // adds r0, :upper0_7:S
  write16(ctx, buf + 8, 0x0200);  // lsls r0, r0, #8
  write16(ctx, buf + 10, 0x3000); // adds r0, :lower8_15:S
  write16(ctx, buf + 12, 0x0200); // lsls r0, r0, #8
  write16(ctx, buf + 14, 0x3000); // adds r0, :lower0_7:S
  write16(ctx, buf + 16, 0x9001); // str r0, [sp, #4] ; SP + 4 = S
  write16(ctx, buf + 18,
          0xbd01); // pop {r0, pc} ; restore r0 and branch to dest
  uint64_t s = getARMThunkDestVA(ctx, destination);
  ctx.target->relocateNoSym(buf + 2, R_ARM_THM_ALU_ABS_G3, s);
  ctx.target->relocateNoSym(buf + 6, R_ARM_THM_ALU_ABS_G2_NC, s);
  ctx.target->relocateNoSym(buf + 10, R_ARM_THM_ALU_ABS_G1_NC, s);
  ctx.target->relocateNoSym(buf + 14, R_ARM_THM_ALU_ABS_G0_NC, s);
}

void ThumbV6MABSXOLongThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__Thumbv6MABSXOLongThunk_" + destination.getName()),
            STT_FUNC, 1, isec);
  addSymbol("$t", STT_NOTYPE, 0, isec);
}

void ThumbV6MPILongThunk::writeLong(uint8_t *buf) {
  // Most Thumb instructions cannot access the high registers r8 - r15. As the
  // only register we can corrupt is ip (r12) we must instead spill a low
  // register to the stack to use as a scratch register.
  write16(ctx, buf + 0,
          0xb401); // P:  push {r0}        ; Obtain scratch register
  write16(ctx, buf + 2, 0x4802); //     ldr r0, [pc, #8] ; L2
  write16(ctx, buf + 4, 0x4684); //     mov ip, r0       ; high to low register
  write16(ctx, buf + 6,
          0xbc01); //     pop {r0}         ; restore scratch register
  write16(ctx, buf + 8, 0x44e7); // L1: add pc, ip       ; transfer control
  write16(ctx, buf + 10,
          0x46c0); //     nop              ; pad to 4-byte boundary
  write32(ctx, buf + 12, 0x00000000); // L2: .word S - (P + (L1 - P) + 4)
  uint64_t s = getARMThunkDestVA(ctx, destination);
  uint64_t p = getThunkTargetSym()->getVA(ctx) & ~0x1;
  ctx.target->relocateNoSym(buf + 12, R_ARM_REL32, s - p - 12);
}

void ThumbV6MPILongThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__Thumbv6MPILongThunk_" + destination.getName()),
            STT_FUNC, 1, isec);
  addSymbol("$t", STT_NOTYPE, 0, isec);
  tsec = &isec;
  (void)getMayUseShortThunk();
}

void ThumbV6MPILongThunk::addLongMapSyms() {
  addSymbol("$d", STT_NOTYPE, 12, *tsec);
}

void ARMV5LongLdrPcThunk::writeLong(uint8_t *buf) {
  write32(ctx, buf + 0, 0xe51ff004); // ldr pc, [pc,#-4] ; L1
  write32(ctx, buf + 4, 0x00000000); // L1: .word S
  ctx.target->relocateNoSym(buf + 4, R_ARM_ABS32,
                            getARMThunkDestVA(ctx, destination));
}

void ARMV5LongLdrPcThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__ARMv5LongLdrPcThunk_" + destination.getName()),
            STT_FUNC, 0, isec);
  addSymbol("$a", STT_NOTYPE, 0, isec);
  tsec = &isec;
  (void)getMayUseShortThunk();
}

void ARMV5LongLdrPcThunk::addLongMapSyms() {
  addSymbol("$d", STT_NOTYPE, 4, *tsec);
}

void ARMV4ABSLongBXThunk::writeLong(uint8_t *buf) {
  write32(ctx, buf + 0, 0xe59fc000); // ldr r12, [pc] ; L1
  write32(ctx, buf + 4, 0xe12fff1c); // bx r12
  write32(ctx, buf + 8, 0x00000000); // L1: .word S
  ctx.target->relocateNoSym(buf + 8, R_ARM_ABS32,
                            getARMThunkDestVA(ctx, destination));
}

void ARMV4ABSLongBXThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__ARMv4ABSLongBXThunk_" + destination.getName()),
            STT_FUNC, 0, isec);
  addSymbol("$a", STT_NOTYPE, 0, isec);
  tsec = &isec;
  (void)getMayUseShortThunk();
}

void ARMV4ABSLongBXThunk::addLongMapSyms() {
  addSymbol("$d", STT_NOTYPE, 8, *tsec);
}

void ThumbV4ABSLongBXThunk::writeLong(uint8_t *buf) {
  write16(ctx, buf + 0, 0x4778); // bx pc
  write16(ctx, buf + 2,
          0xe7fd); // b #-6 ; Arm recommended sequence to follow bx pc
  write32(ctx, buf + 4, 0xe51ff004); // ldr pc, [pc, #-4] ; L1
  write32(ctx, buf + 8, 0x00000000); // L1: .word S
  ctx.target->relocateNoSym(buf + 8, R_ARM_ABS32,
                            getARMThunkDestVA(ctx, destination));
}

void ThumbV4ABSLongBXThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__Thumbv4ABSLongBXThunk_" + destination.getName()),
            STT_FUNC, 1, isec);
  addSymbol("$t", STT_NOTYPE, 0, isec);
  tsec = &isec;
  (void)getMayUseShortThunk();
}

void ThumbV4ABSLongBXThunk::addLongMapSyms() {
  addSymbol("$a", STT_NOTYPE, 4, *tsec);
  addSymbol("$d", STT_NOTYPE, 8, *tsec);
}

void ThumbV4ABSLongThunk::writeLong(uint8_t *buf) {
  write16(ctx, buf + 0, 0x4778); // bx pc
  write16(ctx, buf + 2,
          0xe7fd); // b #-6 ; Arm recommended sequence to follow bx pc
  write32(ctx, buf + 4, 0xe59fc000);  // ldr r12, [pc] ; L1
  write32(ctx, buf + 8, 0xe12fff1c);  // bx r12
  write32(ctx, buf + 12, 0x00000000); // L1: .word S
  ctx.target->relocateNoSym(buf + 12, R_ARM_ABS32,
                            getARMThunkDestVA(ctx, destination));
}

void ThumbV4ABSLongThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__Thumbv4ABSLongThunk_" + destination.getName()),
            STT_FUNC, 1, isec);
  addSymbol("$t", STT_NOTYPE, 0, isec);
  tsec = &isec;
  (void)getMayUseShortThunk();
}

void ThumbV4ABSLongThunk::addLongMapSyms() {
  addSymbol("$a", STT_NOTYPE, 4, *tsec);
  addSymbol("$d", STT_NOTYPE, 12, *tsec);
}

void ARMV4PILongBXThunk::writeLong(uint8_t *buf) {
  write32(ctx, buf + 0, 0xe59fc004);  // P:  ldr ip, [pc,#4] ; L2
  write32(ctx, buf + 4, 0xe08fc00c);  // L1: add ip, pc, ip
  write32(ctx, buf + 8, 0xe12fff1c);  //     bx ip
  write32(ctx, buf + 12, 0x00000000); // L2: .word S - (P + (L1 - P) + 8)
  uint64_t s = getARMThunkDestVA(ctx, destination);
  uint64_t p = getThunkTargetSym()->getVA(ctx) & ~0x1;
  ctx.target->relocateNoSym(buf + 12, R_ARM_REL32, s - p - 12);
}

void ARMV4PILongBXThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__ARMv4PILongBXThunk_" + destination.getName()),
            STT_FUNC, 0, isec);
  addSymbol("$a", STT_NOTYPE, 0, isec);
  tsec = &isec;
  (void)getMayUseShortThunk();
}

void ARMV4PILongBXThunk::addLongMapSyms() {
  addSymbol("$d", STT_NOTYPE, 12, *tsec);
}

void ARMV4PILongThunk::writeLong(uint8_t *buf) {
  write32(ctx, buf + 0, 0xe59fc000); // P:  ldr ip, [pc] ; L2
  write32(ctx, buf + 4, 0xe08ff00c); // L1: add pc, pc, r12
  write32(ctx, buf + 8, 0x00000000); // L2: .word S - (P + (L1 - P) + 8)
  uint64_t s = getARMThunkDestVA(ctx, destination);
  uint64_t p = getThunkTargetSym()->getVA(ctx) & ~0x1;
  ctx.target->relocateNoSym(buf + 8, R_ARM_REL32, s - p - 12);
}

void ARMV4PILongThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__ARMv4PILongThunk_" + destination.getName()),
            STT_FUNC, 0, isec);
  addSymbol("$a", STT_NOTYPE, 0, isec);
  tsec = &isec;
  (void)getMayUseShortThunk();
}

void ARMV4PILongThunk::addLongMapSyms() {
  addSymbol("$d", STT_NOTYPE, 8, *tsec);
}

void ThumbV4PILongBXThunk::writeLong(uint8_t *buf) {
  write16(ctx, buf + 0, 0x4778); // P:  bx pc
  write16(ctx, buf + 2,
          0xe7fd); //     b #-6 ; Arm recommended sequence to follow bx pc
  write32(ctx, buf + 4, 0xe59fc000);  //     ldr r12, [pc] ; L2
  write32(ctx, buf + 8, 0xe08cf00f);  // L1: add pc, r12, pc
  write32(ctx, buf + 12, 0x00000000); // L2: .word S - (P + (L1 - P) + 8)
  uint64_t s = getARMThunkDestVA(ctx, destination);
  uint64_t p = getThunkTargetSym()->getVA(ctx) & ~0x1;
  ctx.target->relocateNoSym(buf + 12, R_ARM_REL32, s - p - 16);
}

void ThumbV4PILongBXThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__Thumbv4PILongBXThunk_" + destination.getName()),
            STT_FUNC, 1, isec);
  addSymbol("$t", STT_NOTYPE, 0, isec);
  tsec = &isec;
  (void)getMayUseShortThunk();
}

void ThumbV4PILongBXThunk::addLongMapSyms() {
  addSymbol("$a", STT_NOTYPE, 4, *tsec);
  addSymbol("$d", STT_NOTYPE, 12, *tsec);
}

void ThumbV4PILongThunk::writeLong(uint8_t *buf) {
  write16(ctx, buf + 0, 0x4778); // P:  bx pc
  write16(ctx, buf + 2,
          0xe7fd); //     b #-6 ; Arm recommended sequence to follow bx pc
  write32(ctx, buf + 4, 0xe59fc004);  //     ldr ip, [pc,#4] ; L2
  write32(ctx, buf + 8, 0xe08fc00c);  // L1: add ip, pc, ip
  write32(ctx, buf + 12, 0xe12fff1c); //     bx ip
  write32(ctx, buf + 16, 0x00000000); // L2: .word S - (P + (L1 - P) + 8)
  uint64_t s = getARMThunkDestVA(ctx, destination);
  uint64_t p = getThunkTargetSym()->getVA(ctx) & ~0x1;
  ctx.target->relocateNoSym(buf + 16, R_ARM_REL32, s - p - 16);
}

void ThumbV4PILongThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__Thumbv4PILongThunk_" + destination.getName()),
            STT_FUNC, 1, isec);
  addSymbol("$t", STT_NOTYPE, 0, isec);
  tsec = &isec;
  (void)getMayUseShortThunk();
}

void ThumbV4PILongThunk::addLongMapSyms() {
  addSymbol("$a", STT_NOTYPE, 4, *tsec);
  addSymbol("$d", STT_NOTYPE, 16, *tsec);
}

// Use the long jump which covers a range up to 8MiB.
void AVRThunk::writeTo(uint8_t *buf) {
  write32(ctx, buf, 0x940c); // jmp func
  ctx.target->relocateNoSym(buf, R_AVR_CALL, destination.getVA(ctx));
}

void AVRThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__AVRThunk_" + destination.getName()), STT_FUNC, 0,
            isec);
}

// Write MIPS LA25 thunk code to call PIC function from the non-PIC one.
void MipsThunk::writeTo(uint8_t *buf) {
  uint64_t s = destination.getVA(ctx);
  write32(ctx, buf, 0x3c190000);                // lui   $25, %hi(func)
  write32(ctx, buf + 4, 0x08000000 | (s >> 2)); // j     func
  write32(ctx, buf + 8, 0x27390000);            // addiu $25, $25, %lo(func)
  write32(ctx, buf + 12, 0x00000000);           // nop
  ctx.target->relocateNoSym(buf, R_MIPS_HI16, s);
  ctx.target->relocateNoSym(buf + 8, R_MIPS_LO16, s);
}

void MipsThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__LA25Thunk_" + destination.getName()), STT_FUNC, 0,
            isec);
}

InputSection *MipsThunk::getTargetInputSection() const {
  auto &dr = cast<Defined>(destination);
  return dyn_cast<InputSection>(dr.section);
}

// Write microMIPS R2-R5 LA25 thunk code
// to call PIC function from the non-PIC one.
void MicroMipsThunk::writeTo(uint8_t *buf) {
  uint64_t s = destination.getVA(ctx);
  write16(ctx, buf, 0x41b9);      // lui   $25, %hi(func)
  write16(ctx, buf + 4, 0xd400);  // j     func
  write16(ctx, buf + 8, 0x3339);  // addiu $25, $25, %lo(func)
  write16(ctx, buf + 12, 0x0c00); // nop
  ctx.target->relocateNoSym(buf, R_MICROMIPS_HI16, s);
  ctx.target->relocateNoSym(buf + 4, R_MICROMIPS_26_S1, s);
  ctx.target->relocateNoSym(buf + 8, R_MICROMIPS_LO16, s);
}

void MicroMipsThunk::addSymbols(ThunkSection &isec) {
  Defined *d =
      addSymbol(ctx.saver.save("__microLA25Thunk_" + destination.getName()),
                STT_FUNC, 0, isec);
  d->stOther |= STO_MIPS_MICROMIPS;
}

InputSection *MicroMipsThunk::getTargetInputSection() const {
  auto &dr = cast<Defined>(destination);
  return dyn_cast<InputSection>(dr.section);
}

// Write microMIPS R6 LA25 thunk code
// to call PIC function from the non-PIC one.
void MicroMipsR6Thunk::writeTo(uint8_t *buf) {
  uint64_t s = destination.getVA(ctx);
  uint64_t p = getThunkTargetSym()->getVA(ctx);
  write16(ctx, buf, 0x1320);     // lui   $25, %hi(func)
  write16(ctx, buf + 4, 0x3339); // addiu $25, $25, %lo(func)
  write16(ctx, buf + 8, 0x9400); // bc    func
  ctx.target->relocateNoSym(buf, R_MICROMIPS_HI16, s);
  ctx.target->relocateNoSym(buf + 4, R_MICROMIPS_LO16, s);
  ctx.target->relocateNoSym(buf + 8, R_MICROMIPS_PC26_S1, s - p - 12);
}

void MicroMipsR6Thunk::addSymbols(ThunkSection &isec) {
  Defined *d =
      addSymbol(ctx.saver.save("__microLA25Thunk_" + destination.getName()),
                STT_FUNC, 0, isec);
  d->stOther |= STO_MIPS_MICROMIPS;
}

InputSection *MicroMipsR6Thunk::getTargetInputSection() const {
  auto &dr = cast<Defined>(destination);
  return dyn_cast<InputSection>(dr.section);
}

void elf::writePPC32PltCallStub(Ctx &ctx, uint8_t *buf, uint64_t gotPltVA,
                                const InputFile *file, int64_t addend) {
  if (!ctx.arg.isPic) {
    write32(ctx, buf + 0, 0x3d600000 | (gotPltVA + 0x8000) >> 16); // lis r11,ha
    write32(ctx, buf + 4, 0x816b0000 | (uint16_t)gotPltVA); // lwz r11,l(r11)
    write32(ctx, buf + 8, 0x7d6903a6);                      // mtctr r11
    write32(ctx, buf + 12, 0x4e800420);                     // bctr
    return;
  }
  uint32_t offset;
  if (addend >= 0x8000) {
    // The stub loads an address relative to r30 (.got2+Addend). Addend is
    // almost always 0x8000. The address of .got2 is different in another object
    // file, so a stub cannot be shared.
    offset = gotPltVA -
             (ctx.in.ppc32Got2->getParent()->getVA() +
              (file->ppc32Got2 ? file->ppc32Got2->outSecOff : 0) + addend);
  } else {
    // The stub loads an address relative to _GLOBAL_OFFSET_TABLE_ (which is
    // currently the address of .got).
    offset = gotPltVA - ctx.in.got->getVA();
  }
  uint16_t ha = (offset + 0x8000) >> 16, l = (uint16_t)offset;
  if (ha == 0) {
    write32(ctx, buf + 0, 0x817e0000 | l); // lwz r11,l(r30)
    write32(ctx, buf + 4, 0x7d6903a6);     // mtctr r11
    write32(ctx, buf + 8, 0x4e800420);     // bctr
    write32(ctx, buf + 12, 0x60000000);    // nop
  } else {
    write32(ctx, buf + 0, 0x3d7e0000 | ha); // addis r11,r30,ha
    write32(ctx, buf + 4, 0x816b0000 | l);  // lwz r11,l(r11)
    write32(ctx, buf + 8, 0x7d6903a6);      // mtctr r11
    write32(ctx, buf + 12, 0x4e800420);     // bctr
  }
}

void PPC32PltCallStub::writeTo(uint8_t *buf) {
  writePPC32PltCallStub(ctx, buf, destination.getGotPltVA(ctx), file, addend);
}

void PPC32PltCallStub::addSymbols(ThunkSection &isec) {
  std::string buf;
  raw_string_ostream os(buf);
  os << format_hex_no_prefix(addend, 8);
  if (!ctx.arg.isPic)
    os << ".plt_call32.";
  else if (addend >= 0x8000)
    os << ".got2.plt_pic32.";
  else
    os << ".plt_pic32.";
  os << destination.getName();
  addSymbol(ctx.saver.save(buf), STT_FUNC, 0, isec);
}

bool PPC32PltCallStub::isCompatibleWith(const InputSection &isec,
                                        const Relocation &rel) const {
  return !ctx.arg.isPic || (isec.file == file && rel.addend == addend);
}

void PPC32LongThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__LongThunk_" + destination.getName()), STT_FUNC, 0,
            isec);
}

void PPC32LongThunk::writeTo(uint8_t *buf) {
  auto ha = [](uint32_t v) -> uint16_t { return (v + 0x8000) >> 16; };
  auto lo = [](uint32_t v) -> uint16_t { return v; };
  uint32_t d = destination.getVA(ctx, addend);
  if (ctx.arg.isPic) {
    uint32_t off = d - (getThunkTargetSym()->getVA(ctx) + 8);
    write32(ctx, buf + 0, 0x7c0802a6);            // mflr r12,0
    write32(ctx, buf + 4, 0x429f0005);            // bcl r20,r31,.+4
    write32(ctx, buf + 8, 0x7d8802a6);            // mtctr r12
    write32(ctx, buf + 12, 0x3d8c0000 | ha(off)); // addis r12,r12,off@ha
    write32(ctx, buf + 16, 0x398c0000 | lo(off)); // addi r12,r12,off@l
    write32(ctx, buf + 20, 0x7c0803a6);           // mtlr r0
    buf += 24;
  } else {
    write32(ctx, buf + 0, 0x3d800000 | ha(d)); // lis r12,d@ha
    write32(ctx, buf + 4, 0x398c0000 | lo(d)); // addi r12,r12,d@l
    buf += 8;
  }
  write32(ctx, buf + 0, 0x7d8903a6); // mtctr r12
  write32(ctx, buf + 4, 0x4e800420); // bctr
}

void elf::writePPC64LoadAndBranch(Ctx &ctx, uint8_t *buf, int64_t offset) {
  uint16_t offHa = (offset + 0x8000) >> 16;
  uint16_t offLo = offset & 0xffff;

  write32(ctx, buf + 0, 0x3d820000 | offHa); // addis r12, r2, OffHa
  write32(ctx, buf + 4, 0xe98c0000 | offLo); // ld    r12, OffLo(r12)
  write32(ctx, buf + 8, 0x7d8903a6);         // mtctr r12
  write32(ctx, buf + 12, 0x4e800420);        // bctr
}

void PPC64PltCallStub::writeTo(uint8_t *buf) {
  int64_t offset = destination.getGotPltVA(ctx) - getPPC64TocBase(ctx);
  // Save the TOC pointer to the save-slot reserved in the call frame.
  write32(ctx, buf + 0, 0xf8410018); // std     r2,24(r1)
  writePPC64LoadAndBranch(ctx, buf + 4, offset);
}

void PPC64PltCallStub::addSymbols(ThunkSection &isec) {
  Defined *s = addSymbol(ctx.saver.save("__plt_" + destination.getName()),
                         STT_FUNC, 0, isec);
  s->setNeedsTocRestore(true);
  s->file = destination.file;
}

bool PPC64PltCallStub::isCompatibleWith(const InputSection &isec,
                                        const Relocation &rel) const {
  return rel.type == R_PPC64_REL24 || rel.type == R_PPC64_REL14;
}

void PPC64R2SaveStub::writeTo(uint8_t *buf) {
  const int64_t offset = computeOffset();
  write32(ctx, buf + 0, 0xf8410018); // std  r2,24(r1)
  // The branch offset needs to fit in 26 bits.
  if (getMayUseShortThunk()) {
    write32(ctx, buf + 4, 0x48000000 | (offset & 0x03fffffc)); // b    <offset>
  } else if (isInt<34>(offset)) {
    int nextInstOffset;
    uint64_t tocOffset = destination.getVA(ctx) - getPPC64TocBase(ctx);
    if (tocOffset >> 16 > 0) {
      const uint64_t addi = ADDI_R12_TO_R12_NO_DISP | (tocOffset & 0xffff);
      const uint64_t addis =
          ADDIS_R12_TO_R2_NO_DISP | ((tocOffset >> 16) & 0xffff);
      write32(ctx, buf + 4, addis); // addis r12, r2 , top of offset
      write32(ctx, buf + 8, addi);  // addi  r12, r12, bottom of offset
      nextInstOffset = 12;
    } else {
      const uint64_t addi = ADDI_R12_TO_R2_NO_DISP | (tocOffset & 0xffff);
      write32(ctx, buf + 4, addi); // addi r12, r2, offset
      nextInstOffset = 8;
    }
    write32(ctx, buf + nextInstOffset, MTCTR_R12); // mtctr r12
    write32(ctx, buf + nextInstOffset + 4, BCTR);  // bctr
  } else {
    ctx.in.ppc64LongBranchTarget->addEntry(&destination, addend);
    const int64_t offsetFromTOC =
        ctx.in.ppc64LongBranchTarget->getEntryVA(&destination, addend) -
        getPPC64TocBase(ctx);
    writePPC64LoadAndBranch(ctx, buf + 4, offsetFromTOC);
  }
}

void PPC64R2SaveStub::addSymbols(ThunkSection &isec) {
  Defined *s = addSymbol(ctx.saver.save("__toc_save_" + destination.getName()),
                         STT_FUNC, 0, isec);
  s->setNeedsTocRestore(true);
}

bool PPC64R2SaveStub::isCompatibleWith(const InputSection &isec,
                                       const Relocation &rel) const {
  return rel.type == R_PPC64_REL24 || rel.type == R_PPC64_REL14;
}

void PPC64R12SetupStub::writeTo(uint8_t *buf) {
  int64_t offset =
      (gotPlt ? destination.getGotPltVA(ctx) : destination.getVA(ctx)) -
      getThunkTargetSym()->getVA(ctx);
  if (!isInt<34>(offset))
    reportRangeError(ctx, buf, offset, 34, destination,
                     "R12 setup stub offset");

  int nextInstOffset;
  if (ctx.arg.power10Stubs) {
    const uint64_t imm = (((offset >> 16) & 0x3ffff) << 32) | (offset & 0xffff);
    // pld 12, func@plt@pcrel or  paddi r12, 0, func@pcrel
    writePrefixedInst(ctx, buf,
                      (gotPlt ? PLD_R12_NO_DISP : PADDI_R12_NO_DISP) | imm);
    nextInstOffset = 8;
  } else {
    uint32_t off = offset - 8;
    write32(ctx, buf + 0, 0x7d8802a6);  // mflr 12
    write32(ctx, buf + 4, 0x429f0005);  // bcl 20,31,.+4
    write32(ctx, buf + 8, 0x7d6802a6);  // mflr 11
    write32(ctx, buf + 12, 0x7d8803a6); // mtlr 12
    write32(ctx, buf + 16,
            0x3d8b0000 | ((off + 0x8000) >> 16)); // addis 12,11,off@ha
    if (gotPlt)
      write32(ctx, buf + 20, 0xe98c0000 | (off & 0xffff)); // ld 12, off@l(12)
    else
      write32(ctx, buf + 20, 0x398c0000 | (off & 0xffff)); // addi 12,12,off@l
    nextInstOffset = 24;
  }
  write32(ctx, buf + nextInstOffset, MTCTR_R12); // mtctr r12
  write32(ctx, buf + nextInstOffset + 4, BCTR);  // bctr
}

void PPC64R12SetupStub::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save((gotPlt ? "__plt_pcrel_" : "__gep_setup_") +
                           destination.getName()),
            STT_FUNC, 0, isec);
}

bool PPC64R12SetupStub::isCompatibleWith(const InputSection &isec,
                                         const Relocation &rel) const {
  return rel.type == R_PPC64_REL24_NOTOC;
}

void PPC64LongBranchThunk::writeTo(uint8_t *buf) {
  int64_t offset =
      ctx.in.ppc64LongBranchTarget->getEntryVA(&destination, addend) -
      getPPC64TocBase(ctx);
  writePPC64LoadAndBranch(ctx, buf, offset);
}

void PPC64LongBranchThunk::addSymbols(ThunkSection &isec) {
  addSymbol(ctx.saver.save("__long_branch_" + destination.getName()), STT_FUNC,
            0, isec);
}

bool PPC64LongBranchThunk::isCompatibleWith(const InputSection &isec,
                                            const Relocation &rel) const {
  return rel.type == R_PPC64_REL24 || rel.type == R_PPC64_REL14;
}

// Hexagon Target Thunks
static uint64_t getHexagonThunkDestVA(Ctx &ctx, const Symbol &s, int64_t a) {
  uint64_t v = s.isInPlt(ctx) ? s.getPltVA(ctx) : s.getVA(ctx, a);
  return SignExtend64<32>(v);
}

void HexagonThunk::writeTo(uint8_t *buf) {
  uint64_t s = getHexagonThunkDestVA(ctx, destination, addend);
  uint64_t p = getThunkTargetSym()->getVA(ctx);

  if (ctx.arg.isPic) {
    write32(ctx, buf + 0, 0x00004000); // {  immext(#0)
    ctx.target->relocateNoSym(buf, R_HEX_B32_PCREL_X, s - p);
    write32(ctx, buf + 4, 0x6a49c00e); //    r14 = add(pc,##0) }
    ctx.target->relocateNoSym(buf + 4, R_HEX_6_PCREL_X, s - p);

    write32(ctx, buf + 8, 0x528ec000); // {  jumpr r14 }
  } else {
    write32(ctx, buf + 0, 0x00004000); //  { immext
    ctx.target->relocateNoSym(buf, R_HEX_B32_PCREL_X, s - p);
    write32(ctx, buf + 4, 0x5800c000); //    jump <> }
    ctx.target->relocateNoSym(buf + 4, R_HEX_B22_PCREL_X, s - p);
  }
}
void HexagonThunk::addSymbols(ThunkSection &isec) {
  Symbol *enclosing = isec.getEnclosingSymbol(relOffset);
  StringRef src = enclosing ? enclosing->getName() : isec.name;

  addSymbol(
      saver().save("__hexagon_thunk_" + destination.getName() + "_from_" + src),
      STT_FUNC, 0, isec);
}

Thunk::Thunk(Ctx &ctx, Symbol &d, int64_t a)
    : ctx(ctx), destination(d), addend(a), offset(0) {
  destination.thunkAccessed = true;
}

Thunk::~Thunk() = default;

static std::unique_ptr<Thunk> addThunkAArch64(Ctx &ctx, const InputSection &sec,
                                              RelType type, Symbol &s,
                                              int64_t a) {
  assert(is_contained({R_AARCH64_CALL26, R_AARCH64_JUMP26, R_AARCH64_PLT32},
                      type));
  bool mayNeedLandingPad =
      (ctx.arg.andFeatures & GNU_PROPERTY_AARCH64_FEATURE_1_BTI) &&
      !isAArch64BTILandingPad(ctx, s, a);
  if (ctx.arg.picThunk)
    return std::make_unique<AArch64ADRPThunk>(ctx, s, a, mayNeedLandingPad);
  if (sec.getParent()->flags & SHF_AARCH64_PURECODE)
    return std::make_unique<AArch64ABSXOLongThunk>(ctx, s, a,
                                                   mayNeedLandingPad);
  return std::make_unique<AArch64ABSLongThunk>(ctx, s, a, mayNeedLandingPad);
}

// Creates a thunk for long branches or Thumb-ARM interworking.
// Arm Architectures v4t does not support Thumb2 technology, and does not
// support BLX or LDR Arm/Thumb state switching. This means that
// - MOVT and MOVW instructions cannot be used.
// - We can't rewrite BL in place to BLX. We will need thunks.
//
// TODO: use B for short Thumb->Arm thunks instead of LDR (this doesn't work for
//       Arm->Thumb, as in Arm state no BX PC trick; it doesn't switch state).
static std::unique_ptr<Thunk> addThunkArmv4(Ctx &ctx, RelType reloc, Symbol &s,
                                            int64_t a) {
  bool thumb_target = s.getVA(ctx, a) & 1;

  switch (reloc) {
  case R_ARM_PC24:
  case R_ARM_PLT32:
  case R_ARM_JUMP24:
  case R_ARM_CALL:
    if (ctx.arg.picThunk) {
      if (thumb_target)
        return std::make_unique<ARMV4PILongBXThunk>(ctx, s, a);
      return std::make_unique<ARMV4PILongThunk>(ctx, s, a);
    }
    if (thumb_target)
      return std::make_unique<ARMV4ABSLongBXThunk>(ctx, s, a);
    return std::make_unique<ARMV5LongLdrPcThunk>(ctx, s, a);
  case R_ARM_THM_CALL:
    if (ctx.arg.picThunk) {
      if (thumb_target)
        return std::make_unique<ThumbV4PILongThunk>(ctx, s, a);
      return std::make_unique<ThumbV4PILongBXThunk>(ctx, s, a);
    }
    if (thumb_target)
      return std::make_unique<ThumbV4ABSLongThunk>(ctx, s, a);
    return std::make_unique<ThumbV4ABSLongBXThunk>(ctx, s, a);
  }
  Fatal(ctx) << "relocation " << reloc << " to " << &s
             << " not supported for Armv4 or Armv4T target";
  llvm_unreachable("");
}

// Creates a thunk for Thumb-ARM interworking compatible with Armv5 and Armv6.
// Arm Architectures v5 and v6 do not support Thumb2 technology. This means that
// - MOVT and MOVW instructions cannot be used
// - Only Thumb relocation that can generate a Thunk is a BL, this can always
//   be transformed into a BLX
static std::unique_ptr<Thunk> addThunkArmv5v6(Ctx &ctx, RelType reloc,
                                              Symbol &s, int64_t a) {
  switch (reloc) {
  case R_ARM_PC24:
  case R_ARM_PLT32:
  case R_ARM_JUMP24:
  case R_ARM_CALL:
  case R_ARM_THM_CALL:
    if (ctx.arg.picThunk)
      return std::make_unique<ARMV4PILongBXThunk>(ctx, s, a);
    return std::make_unique<ARMV5LongLdrPcThunk>(ctx, s, a);
  }
  Fatal(ctx) << "relocation " << reloc << " to " << &s
             << " not supported for Armv5 or Armv6 targets";
  llvm_unreachable("");
}

// Create a thunk for Thumb long branch on V6-M.
// Arm Architecture v6-M only supports Thumb instructions. This means
// - MOVT and MOVW instructions cannot be used.
// - Only a limited number of instructions can access registers r8 and above
// - No interworking support is needed (all Thumb).
static std::unique_ptr<Thunk> addThunkV6M(Ctx &ctx, const InputSection &isec,
                                          RelType reloc, Symbol &s, int64_t a) {
  const bool isPureCode = isec.getParent()->flags & SHF_ARM_PURECODE;
  switch (reloc) {
  case R_ARM_THM_JUMP19:
  case R_ARM_THM_JUMP24:
  case R_ARM_THM_CALL:
    if (ctx.arg.isPic) {
      if (!isPureCode)
        return std::make_unique<ThumbV6MPILongThunk>(ctx, s, a);

      Fatal(ctx)
          << "relocation " << reloc << " to " << &s
          << " not supported for Armv6-M targets for position independent"
             " and execute only code";
      llvm_unreachable("");
    }
    if (isPureCode)
      return std::make_unique<ThumbV6MABSXOLongThunk>(ctx, s, a);
    return std::make_unique<ThumbV6MABSLongThunk>(ctx, s, a);
  }
  Fatal(ctx) << "relocation " << reloc << " to " << &s
             << " not supported for Armv6-M targets";
  llvm_unreachable("");
}

// Creates a thunk for Thumb-ARM interworking or branch range extension.
static std::unique_ptr<Thunk> addThunkArm(Ctx &ctx, const InputSection &isec,
                                          RelType reloc, Symbol &s, int64_t a) {
  // Decide which Thunk is needed based on:
  // Available instruction set
  // - An Arm Thunk can only be used if Arm state is available.
  // - A Thumb Thunk can only be used if Thumb state is available.
  // - Can only use a Thunk if it uses instructions that the Target supports.
  // Relocation is branch or branch and link
  // - Branch instructions cannot change state, can only select Thunk that
  //   starts in the same state as the caller.
  // - Branch and link relocations can change state, can select Thunks from
  //   either Arm or Thumb.
  // Position independent Thunks if we require position independent code.
  // Execute Only Thunks if the output section is execute only code.

  // Handle architectures that have restrictions on the instructions that they
  // can use in Thunks. The flags below are set by reading the BuildAttributes
  // of the input objects. InputFiles.cpp contains the mapping from ARM
  // architecture to flag.
  if (!ctx.arg.armHasMovtMovw) {
    if (ctx.arg.armJ1J2BranchEncoding)
      return addThunkV6M(ctx, isec, reloc, s, a);
    if (ctx.arg.armHasBlx)
      return addThunkArmv5v6(ctx, reloc, s, a);
    return addThunkArmv4(ctx, reloc, s, a);
  }

  switch (reloc) {
  case R_ARM_PC24:
  case R_ARM_PLT32:
  case R_ARM_JUMP24:
  case R_ARM_CALL:
    if (ctx.arg.picThunk)
      return std::make_unique<ARMV7PILongThunk>(ctx, s, a);
    return std::make_unique<ARMV7ABSLongThunk>(ctx, s, a);
  case R_ARM_THM_JUMP19:
  case R_ARM_THM_JUMP24:
  case R_ARM_THM_CALL:
    if (ctx.arg.picThunk)
      return std::make_unique<ThumbV7PILongThunk>(ctx, s, a);
    return std::make_unique<ThumbV7ABSLongThunk>(ctx, s, a);
  }
  llvm_unreachable("");
}

static std::unique_ptr<Thunk> addThunkAVR(Ctx &ctx, RelType type, Symbol &s,
                                          int64_t a) {
  switch (type) {
  case R_AVR_LO8_LDI_GS:
  case R_AVR_HI8_LDI_GS:
    return std::make_unique<AVRThunk>(ctx, s, a);
  default:
    llvm_unreachable("");
  }
}

static std::unique_ptr<Thunk> addThunkHexagon(Ctx &ctx,
                                              const InputSection &isec,
                                              Relocation &rel, Symbol &s) {
  switch (rel.type) {
  case R_HEX_B9_PCREL:
  case R_HEX_B13_PCREL:
  case R_HEX_B15_PCREL:
  case R_HEX_B22_PCREL:
  case R_HEX_PLT_B22_PCREL:
  case R_HEX_GD_PLT_B22_PCREL:
    return std::make_unique<HexagonThunk>(ctx, isec, rel, s);
  default:
    Fatal(ctx) << "unrecognized relocation " << rel.type << " to " << &s
               << " for hexagon target";
    llvm_unreachable("");
  }
}

static std::unique_ptr<Thunk> addThunkMips(Ctx &ctx, RelType type, Symbol &s) {
  if ((s.stOther & STO_MIPS_MICROMIPS) && isMipsR6(ctx))
    return std::make_unique<MicroMipsR6Thunk>(ctx, s);
  if (s.stOther & STO_MIPS_MICROMIPS)
    return std::make_unique<MicroMipsThunk>(ctx, s);
  return std::make_unique<MipsThunk>(ctx, s);
}

static std::unique_ptr<Thunk> addThunkPPC32(Ctx &ctx, const InputSection &isec,
                                            const Relocation &rel, Symbol &s) {
  assert((rel.type == R_PPC_LOCAL24PC || rel.type == R_PPC_REL24 ||
          rel.type == R_PPC_PLTREL24) &&
         "unexpected relocation type for thunk");
  if (s.isInPlt(ctx))
    return std::make_unique<PPC32PltCallStub>(ctx, isec, rel, s);
  return std::make_unique<PPC32LongThunk>(ctx, s, rel.addend);
}

static std::unique_ptr<Thunk> addThunkPPC64(Ctx &ctx, RelType type, Symbol &s,
                                            int64_t a) {
  assert((type == R_PPC64_REL14 || type == R_PPC64_REL24 ||
          type == R_PPC64_REL24_NOTOC) &&
         "unexpected relocation type for thunk");

  // If we are emitting stubs for NOTOC relocations, we need to tell
  // the PLT resolver that there can be multiple TOCs.
  if (type == R_PPC64_REL24_NOTOC)
    ctx.target->ppc64DynamicSectionOpt = 0x2;

  if (s.isInPlt(ctx)) {
    if (type == R_PPC64_REL24_NOTOC)
      return std::make_unique<PPC64R12SetupStub>(ctx, s,
                                                 /*gotPlt=*/true);
    return std::make_unique<PPC64PltCallStub>(ctx, s);
  }

  // This check looks at the st_other bits of the callee. If the value is 1
  // then the callee clobbers the TOC and we need an R2 save stub when RelType
  // is R_PPC64_REL14 or R_PPC64_REL24.
  if ((type == R_PPC64_REL14 || type == R_PPC64_REL24) && (s.stOther >> 5) == 1)
    return std::make_unique<PPC64R2SaveStub>(ctx, s, a);

  if (type == R_PPC64_REL24_NOTOC)
    return std::make_unique<PPC64R12SetupStub>(ctx, s, /*gotPlt=*/false);

  if (ctx.arg.picThunk)
    return std::make_unique<PPC64PILongBranchThunk>(ctx, s, a);

  return std::make_unique<PPC64PDLongBranchThunk>(ctx, s, a);
}

std::unique_ptr<Thunk> elf::addThunk(Ctx &ctx, const InputSection &isec,
                                     Relocation &rel) {
  Symbol &s = *rel.sym;
  int64_t a = rel.addend;

  switch (ctx.arg.emachine) {
  case EM_AARCH64:
    return addThunkAArch64(ctx, isec, rel.type, s, a);
  case EM_ARM:
    return addThunkArm(ctx, isec, rel.type, s, a);
  case EM_AVR:
    return addThunkAVR(ctx, rel.type, s, a);
  case EM_MIPS:
    return addThunkMips(ctx, rel.type, s);
  case EM_PPC:
    return addThunkPPC32(ctx, isec, rel, s);
  case EM_PPC64:
    return addThunkPPC64(ctx, rel.type, s, a);
  case EM_HEXAGON:
    return addThunkHexagon(ctx, isec, rel, s);
  default:
    llvm_unreachable(
        "add Thunk only supported for ARM, AVR, Hexagon, Mips and PowerPC");
  }
}

std::unique_ptr<Thunk> elf::addLandingPadThunk(Ctx &ctx, Symbol &s, int64_t a) {
  switch (ctx.arg.emachine) {
  case EM_AARCH64:
    return std::make_unique<AArch64BTILandingPadThunk>(ctx, s, a);
  default:
    llvm_unreachable("add landing pad only supported for AArch64");
  }
}
