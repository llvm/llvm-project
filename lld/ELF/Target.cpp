//===- Target.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Machine-specific things, such as applying relocations, creation of
// GOT or PLT entries, etc., are handled in this file.
//
// Refer the ELF spec for the single letter varaibles, S, A or P, used
// in this file. SA is S+A.
//
//===----------------------------------------------------------------------===//

#include "Target.h"
#include "Error.h"
#include "OutputSections.h"
#include "Symbols.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ELF.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;

namespace lld {
namespace elf2 {

std::unique_ptr<TargetInfo> Target;

template <endianness E> static void add32(void *P, int32_t V) {
  write32<E>(P, read32<E>(P) + V);
}

static void add32le(uint8_t *P, int32_t V) { add32<support::little>(P, V); }
static void or32le(uint8_t *P, int32_t V) { write32le(P, read32le(P) | V); }

template <unsigned N> static void checkInt(int64_t V, uint32_t Type) {
  if (isInt<N>(V))
    return;
  StringRef S = getELFRelocationTypeName(Config->EMachine, Type);
  error("Relocation " + S + " out of range");
}

template <unsigned N> static void checkUInt(uint64_t V, uint32_t Type) {
  if (isUInt<N>(V))
    return;
  StringRef S = getELFRelocationTypeName(Config->EMachine, Type);
  error("Relocation " + S + " out of range");
}

template <unsigned N> static void checkIntUInt(uint64_t V, uint32_t Type) {
  if (isInt<N>(V) || isUInt<N>(V))
    return;
  StringRef S = getELFRelocationTypeName(Config->EMachine, Type);
  error("Relocation " + S + " out of range");
}

template <unsigned N> static void checkAlignment(uint64_t V, uint32_t Type) {
  if ((V & (N - 1)) == 0)
    return;
  StringRef S = getELFRelocationTypeName(Config->EMachine, Type);
  error("Improper alignment for relocation " + S);
}

namespace {
class X86TargetInfo final : public TargetInfo {
public:
  X86TargetInfo();
  void writeGotPltHeaderEntries(uint8_t *Buf) const override;
  unsigned getDynReloc(unsigned Type) const override;
  bool isTlsDynReloc(unsigned Type) const override;
  void writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const override;
  void writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                         uint64_t PltEntryAddr) const override;
  void writePltEntry(uint8_t *Buf, uint64_t GotAddr, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr, int32_t Index,
                     unsigned RelOff) const override;
  bool relocNeedsCopy(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Loc, uint8_t *BufEnd, uint32_t Type, uint64_t P,
                   uint64_t SA) const override;
};

class X86_64TargetInfo final : public TargetInfo {
public:
  X86_64TargetInfo();
  unsigned getPltRefReloc(unsigned Type) const override;
  bool isTlsDynReloc(unsigned Type) const override;
  void writeGotPltHeaderEntries(uint8_t *Buf) const override;
  void writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const override;
  void writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                         uint64_t PltEntryAddr) const override;
  void writePltEntry(uint8_t *Buf, uint64_t GotAddr, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr, int32_t Index,
                     unsigned RelOff) const override;
  bool relocNeedsCopy(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Loc, uint8_t *BufEnd, uint32_t Type, uint64_t P,
                   uint64_t SA) const override;
  bool isRelRelative(uint32_t Type) const override;
  bool isTlsOptimized(unsigned Type, const SymbolBody *S) const override;
  unsigned relocateTlsOptimize(uint8_t *Loc, uint8_t *BufEnd, uint32_t Type,
                               uint64_t P, uint64_t SA) const override;

private:
  void relocateTlsLdToLe(uint8_t *Loc, uint8_t *BufEnd, uint64_t P,
                         uint64_t SA) const;
  void relocateTlsGdToLe(uint8_t *Loc, uint8_t *BufEnd, uint64_t P,
                         uint64_t SA) const;
  void relocateTlsIeToLe(uint8_t *Loc, uint8_t *BufEnd, uint64_t P,
                         uint64_t SA) const;
};

class PPC64TargetInfo final : public TargetInfo {
public:
  PPC64TargetInfo();
  void writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const override;
  void writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                         uint64_t PltEntryAddr) const override;
  void writePltEntry(uint8_t *Buf, uint64_t GotAddr, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr, int32_t Index,
                     unsigned RelOff) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Loc, uint8_t *BufEnd, uint32_t Type, uint64_t P,
                   uint64_t SA) const override;
  bool isRelRelative(uint32_t Type) const override;
};

class AArch64TargetInfo final : public TargetInfo {
public:
  AArch64TargetInfo();
  unsigned getGotRefReloc(unsigned Type) const override;
  unsigned getPltRefReloc(unsigned Type) const override;
  void writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const override;
  void writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                         uint64_t PltEntryAddr) const override;
  void writePltEntry(uint8_t *Buf, uint64_t GotAddr, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr, int32_t Index,
                     unsigned RelOff) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Loc, uint8_t *BufEnd, uint32_t Type, uint64_t P,
                   uint64_t SA) const override;
};

template <class ELFT> class MipsTargetInfo final : public TargetInfo {
public:
  MipsTargetInfo();
  unsigned getGotRefReloc(unsigned Type) const override;
  void writeGotHeaderEntries(uint8_t *Buf) const override;
  void writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const override;
  void writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                         uint64_t PltEntryAddr) const override;
  void writePltEntry(uint8_t *Buf, uint64_t GotAddr, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr, int32_t Index,
                     unsigned RelOff) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Loc, uint8_t *BufEnd, uint32_t Type, uint64_t P,
                   uint64_t SA) const override;
};
} // anonymous namespace

TargetInfo *createTarget() {
  switch (Config->EMachine) {
  case EM_386:
    return new X86TargetInfo();
  case EM_AARCH64:
    return new AArch64TargetInfo();
  case EM_MIPS:
    switch (Config->EKind) {
    case ELF32LEKind:
      return new MipsTargetInfo<ELF32LE>();
    case ELF32BEKind:
      return new MipsTargetInfo<ELF32BE>();
    default:
      error("Unsupported MIPS target");
    }
  case EM_PPC64:
    return new PPC64TargetInfo();
  case EM_X86_64:
    return new X86_64TargetInfo();
  }
  error("Unknown target machine");
}

TargetInfo::~TargetInfo() {}

bool TargetInfo::isTlsOptimized(unsigned Type, const SymbolBody *S) const {
  return false;
}

uint64_t TargetInfo::getVAStart() const { return Config->Shared ? 0 : VAStart; }

bool TargetInfo::relocNeedsCopy(uint32_t Type, const SymbolBody &S) const {
  return false;
}

unsigned TargetInfo::getGotRefReloc(unsigned Type) const { return GotRefReloc; }

unsigned TargetInfo::getPltRefReloc(unsigned Type) const { return PCRelReloc; }

bool TargetInfo::isRelRelative(uint32_t Type) const { return true; }

unsigned TargetInfo::relocateTlsOptimize(uint8_t *Loc, uint8_t *BufEnd,
                                         uint32_t Type, uint64_t P,
                                         uint64_t SA) const {
  return 0;
}

void TargetInfo::writeGotHeaderEntries(uint8_t *Buf) const {}

void TargetInfo::writeGotPltHeaderEntries(uint8_t *Buf) const {}

X86TargetInfo::X86TargetInfo() {
  CopyReloc = R_386_COPY;
  PCRelReloc = R_386_PC32;
  GotReloc = R_386_GLOB_DAT;
  GotRefReloc = R_386_GOT32;
  PltReloc = R_386_JUMP_SLOT;
  LazyRelocations = true;
  PltEntrySize = 16;
  PltZeroEntrySize = 16;
}

void X86TargetInfo::writeGotPltHeaderEntries(uint8_t *Buf) const {
  write32le(Buf, Out<ELF32LE>::Dynamic->getVA());
}

void X86TargetInfo::writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const {
  // Skip 6 bytes of "pushl (GOT+4)"
  write32le(Buf, Plt + 6);
}

unsigned X86TargetInfo::getDynReloc(unsigned Type) const {
  if (Type == R_386_TLS_LE)
    return R_386_TLS_TPOFF;
  if (Type == R_386_TLS_LE_32)
    return R_386_TLS_TPOFF32;
  return Type;
}

bool X86TargetInfo::isTlsDynReloc(unsigned Type) const {
  if (Type == R_386_TLS_LE || Type == R_386_TLS_LE_32)
    return Config->Shared;
  return false;
}

void X86TargetInfo::writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                      uint64_t PltEntryAddr) const {
  // Executable files and shared object files have
  // separate procedure linkage tables.
  if (Config->Shared) {
    const uint8_t V[] = {
        0xff, 0xb3, 0x04, 0x00, 0x00, 0x00, // pushl 4(%ebx
        0xff, 0xa3, 0x08, 0x00, 0x00, 0x00, // jmp *8(%ebx)
        0x90, 0x90, 0x90, 0x90              // nop;nop;nop;nop
    };
    memcpy(Buf, V, sizeof(V));
    return;
  }

  const uint8_t PltData[] = {
      0xff, 0x35, 0x00, 0x00, 0x00, 0x00, // pushl (GOT+4)
      0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // jmp *(GOT+8)
      0x90, 0x90, 0x90, 0x90              // nop;nop;nop;nop
  };
  memcpy(Buf, PltData, sizeof(PltData));
  write32le(Buf + 2, GotEntryAddr + 4); // GOT+4
  write32le(Buf + 8, GotEntryAddr + 8); // GOT+8
}

void X86TargetInfo::writePltEntry(uint8_t *Buf, uint64_t GotAddr,
                                  uint64_t GotEntryAddr, uint64_t PltEntryAddr,
                                  int32_t Index, unsigned RelOff) const {
  const uint8_t Inst[] = {
      0xff, 0x00, 0x00, 0x00, 0x00, 0x00, // jmp *foo_in_GOT|*foo@GOT(%ebx)
      0x68, 0x00, 0x00, 0x00, 0x00,       // pushl $reloc_offset
      0xe9, 0x00, 0x00, 0x00, 0x00        // jmp .PLT0@PC
  };
  memcpy(Buf, Inst, sizeof(Inst));
  // jmp *foo@GOT(%ebx) or jmp *foo_in_GOT
  Buf[1] = Config->Shared ? 0xa3 : 0x25;
  write32le(Buf + 2, Config->Shared ? (GotEntryAddr - GotAddr) : GotEntryAddr);
  write32le(Buf + 7, RelOff);
  write32le(Buf + 12, -Index * PltEntrySize - PltZeroEntrySize - 16);
}

bool X86TargetInfo::relocNeedsCopy(uint32_t Type, const SymbolBody &S) const {
  if (Type == R_386_32 || Type == R_386_16 || Type == R_386_8)
    if (auto *SS = dyn_cast<SharedSymbol<ELF32LE>>(&S))
      return SS->Sym.getType() == STT_OBJECT;
  return false;
}

bool X86TargetInfo::relocNeedsGot(uint32_t Type, const SymbolBody &S) const {
  return Type == R_386_GOT32 || relocNeedsPlt(Type, S);
}

bool X86TargetInfo::relocNeedsPlt(uint32_t Type, const SymbolBody &S) const {
  return Type == R_386_PLT32 || (Type == R_386_PC32 && S.isShared());
}

void X86TargetInfo::relocateOne(uint8_t *Loc, uint8_t *BufEnd, uint32_t Type,
                                uint64_t P, uint64_t SA) const {
  switch (Type) {
  case R_386_GOT32:
    add32le(Loc, SA - Out<ELF32LE>::Got->getVA());
    break;
  case R_386_GOTPC:
    add32le(Loc, SA + Out<ELF32LE>::Got->getVA() - P);
    break;
  case R_386_PC32:
    add32le(Loc, SA - P);
    break;
  case R_386_32:
    add32le(Loc, SA);
    break;
  case R_386_TLS_LE:
    write32le(Loc, SA - Out<ELF32LE>::TlsPhdr->p_memsz);
    break;
  case R_386_TLS_LE_32:
    write32le(Loc, Out<ELF32LE>::TlsPhdr->p_memsz - SA);
    break;
  default:
    error("unrecognized reloc " + Twine(Type));
  }
}

X86_64TargetInfo::X86_64TargetInfo() {
  CopyReloc = R_X86_64_COPY;
  PCRelReloc = R_X86_64_PC32;
  GotReloc = R_X86_64_GLOB_DAT;
  GotRefReloc = R_X86_64_PC32;
  PltReloc = R_X86_64_JUMP_SLOT;
  RelativeReloc = R_X86_64_RELATIVE;
  TlsGotReloc = R_X86_64_TPOFF64;
  TlsLocalDynamicReloc = R_X86_64_TLSLD;
  TlsGlobalDynamicReloc = R_X86_64_TLSGD;
  TlsModuleIndexReloc = R_X86_64_DTPMOD64;
  TlsOffsetReloc = R_X86_64_DTPOFF64;
  LazyRelocations = true;
  PltEntrySize = 16;
  PltZeroEntrySize = 16;
}

void X86_64TargetInfo::writeGotPltHeaderEntries(uint8_t *Buf) const {
  write64le(Buf, Out<ELF64LE>::Dynamic->getVA());
}

void X86_64TargetInfo::writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const {
  // Skip 6 bytes of "jmpq *got(%rip)"
  write32le(Buf, Plt + 6);
}

void X86_64TargetInfo::writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                         uint64_t PltEntryAddr) const {
  const uint8_t PltData[] = {
      0xff, 0x35, 0x00, 0x00, 0x00, 0x00, // pushq GOT+8(%rip)
      0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // jmp *GOT+16(%rip)
      0x0f, 0x1f, 0x40, 0x00              // nopl 0x0(rax)
  };
  memcpy(Buf, PltData, sizeof(PltData));
  write32le(Buf + 2, GotEntryAddr - PltEntryAddr + 2); // GOT+8
  write32le(Buf + 8, GotEntryAddr - PltEntryAddr + 4); // GOT+16
}

void X86_64TargetInfo::writePltEntry(uint8_t *Buf, uint64_t GotAddr,
                                     uint64_t GotEntryAddr,
                                     uint64_t PltEntryAddr, int32_t Index,
                                     unsigned RelOff) const {
  const uint8_t Inst[] = {
      0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // jmpq *got(%rip)
      0x68, 0x00, 0x00, 0x00, 0x00,       // pushq <relocation index>
      0xe9, 0x00, 0x00, 0x00, 0x00        // jmpq plt[0]
  };
  memcpy(Buf, Inst, sizeof(Inst));

  write32le(Buf + 2, GotEntryAddr - PltEntryAddr - 6);
  write32le(Buf + 7, Index);
  write32le(Buf + 12, -Index * PltEntrySize - PltZeroEntrySize - 16);
}

bool X86_64TargetInfo::relocNeedsCopy(uint32_t Type,
                                      const SymbolBody &S) const {
  if (Type == R_X86_64_32S || Type == R_X86_64_32 || Type == R_X86_64_PC32 ||
      Type == R_X86_64_64)
    if (auto *SS = dyn_cast<SharedSymbol<ELF64LE>>(&S))
      return SS->Sym.getType() == STT_OBJECT;
  return false;
}

bool X86_64TargetInfo::relocNeedsGot(uint32_t Type, const SymbolBody &S) const {
  if (Type == R_X86_64_GOTTPOFF)
    return !isTlsOptimized(Type, &S);
  return Type == R_X86_64_GOTTPOFF || Type == R_X86_64_GOTPCREL ||
         relocNeedsPlt(Type, S);
}

bool X86_64TargetInfo::isTlsDynReloc(unsigned Type) const {
  return Type == R_X86_64_GOTTPOFF;
}

unsigned X86_64TargetInfo::getPltRefReloc(unsigned Type) const {
  if (Type == R_X86_64_PLT32)
    return R_X86_64_PC32;
  return Type;
}

bool X86_64TargetInfo::relocNeedsPlt(uint32_t Type, const SymbolBody &S) const {
  if (relocNeedsCopy(Type, S))
    return false;

  switch (Type) {
  default:
    return false;
  case R_X86_64_32:
  case R_X86_64_64:
  case R_X86_64_PC32:
    // This relocation is defined to have a value of (S + A - P).
    // The problems start when a non PIC program calls a function in a shared
    // library.
    // In an ideal world, we could just report an error saying the relocation
    // can overflow at runtime.
    // In the real world with glibc, crt1.o has a R_X86_64_PC32 pointing to
    // libc.so.
    //
    // The general idea on how to handle such cases is to create a PLT entry
    // and use that as the function value.
    //
    // For the static linking part, we just return true and everything else
    // will use the the PLT entry as the address.
    //
    // The remaining (unimplemented) problem is making sure pointer equality
    // still works. We need the help of the dynamic linker for that. We
    // let it know that we have a direct reference to a so symbol by creating
    // an undefined symbol with a non zero st_value. Seeing that, the
    // dynamic linker resolves the symbol to the value of the symbol we created.
    // This is true even for got entries, so pointer equality is maintained.
    // To avoid an infinite loop, the only entry that points to the
    // real function is a dedicated got entry used by the plt. That is
    // identified by special relocation types (R_X86_64_JUMP_SLOT,
    // R_386_JMP_SLOT, etc).
    return S.isShared();
  case R_X86_64_PLT32:
    return canBePreempted(&S, true);
  }
}

bool X86_64TargetInfo::isRelRelative(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case R_X86_64_PC64:
  case R_X86_64_PC32:
  case R_X86_64_PC16:
  case R_X86_64_PC8:
  case R_X86_64_PLT32:
  case R_X86_64_DTPOFF32:
  case R_X86_64_DTPOFF64:
    return true;
  }
}

bool X86_64TargetInfo::isTlsOptimized(unsigned Type,
                                      const SymbolBody *S) const {
  if (Config->Shared || (S && !S->isTLS()))
    return false;
  return Type == R_X86_64_TLSLD || Type == R_X86_64_DTPOFF32 ||
         (Type == R_X86_64_TLSGD && !canBePreempted(S, true)) ||
         (Type == R_X86_64_GOTTPOFF && !canBePreempted(S, true));
}

// "Ulrich Drepper, ELF Handling For Thread-Local Storage" (5.5
// x86-x64 linker optimizations, http://www.akkadia.org/drepper/tls.pdf) shows
// how LD can be optimized to LE:
//   leaq bar@tlsld(%rip), %rdi
//   callq __tls_get_addr@PLT
//   leaq bar@dtpoff(%rax), %rcx
// Is converted to:
//  .word 0x6666
//  .byte 0x66
//  mov %fs:0,%rax
//  leaq bar@tpoff(%rax), %rcx
void X86_64TargetInfo::relocateTlsLdToLe(uint8_t *Loc, uint8_t *BufEnd,
                                         uint64_t P, uint64_t SA) const {
  const uint8_t Inst[] = {
      0x66, 0x66,                                          //.word 0x6666
      0x66,                                                //.byte 0x66
      0x64, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00 // mov %fs:0,%rax
  };
  memcpy(Loc - 3, Inst, sizeof(Inst));
}

// "Ulrich Drepper, ELF Handling For Thread-Local Storage" (5.5
// x86-x64 linker optimizations, http://www.akkadia.org/drepper/tls.pdf) shows
// how GD can be optimized to LE:
//  .byte 0x66
//  leaq x@tlsgd(%rip), %rdi
//  .word 0x6666
//  rex64
//  call __tls_get_addr@plt
// Is converted to:
//  mov %fs:0x0,%rax
//  lea x@tpoff,%rax
void X86_64TargetInfo::relocateTlsGdToLe(uint8_t *Loc, uint8_t *BufEnd,
                                         uint64_t P, uint64_t SA) const {
  const uint8_t Inst[] = {
      0x64, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, // mov %fs:0x0,%rax
      0x48, 0x8d, 0x80, 0x00, 0x00, 0x00, 0x00              // lea x@tpoff,%rax
  };
  memcpy(Loc - 4, Inst, sizeof(Inst));
  relocateOne(Loc + 8, BufEnd, R_X86_64_TPOFF32, P, SA);
}

// In some conditions, R_X86_64_GOTTPOFF relocation can be optimized to
// R_X86_64_TPOFF32 so that R_X86_64_TPOFF32 so that it does not use GOT.
// This function does that. Read "ELF Handling For Thread-Local Storage,
// 5.5 x86-x64 linker optimizations" (http://www.akkadia.org/drepper/tls.pdf)
// by Ulrich Drepper for details.
void X86_64TargetInfo::relocateTlsIeToLe(uint8_t *Loc, uint8_t *BufEnd,
                                         uint64_t P, uint64_t SA) const {
  // Ulrich's document section 6.5 says that @gottpoff(%rip) must be
  // used in MOVQ or ADDQ instructions only.
  // "MOVQ foo@GOTTPOFF(%RIP), %REG" is transformed to "MOVQ $foo, %REG".
  // "ADDQ foo@GOTTPOFF(%RIP), %REG" is transformed to "LEAQ foo(%REG), %REG"
  // (if the register is not RSP/R12) or "ADDQ $foo, %RSP".
  // Opcodes info can be found at http://ref.x86asm.net/coder64.html#x48.
  uint8_t *Prefix = Loc - 3;
  uint8_t *Inst = Loc - 2;
  uint8_t *RegSlot = Loc - 1;
  uint8_t Reg = Loc[-1] >> 3;
  bool IsMov = *Inst == 0x8b;
  bool RspAdd = !IsMov && Reg == 4;
  // r12 and rsp registers requires special handling.
  // Problem is that for other registers, for example leaq 0xXXXXXXXX(%r11),%r11
  // result out is 7 bytes: 4d 8d 9b XX XX XX XX,
  // but leaq 0xXXXXXXXX(%r12),%r12 is 8 bytes: 4d 8d a4 24 XX XX XX XX.
  // The same true for rsp. So we convert to addq for them, saving 1 byte that
  // we dont have.
  if (RspAdd)
    *Inst = 0x81;
  else
    *Inst = IsMov ? 0xc7 : 0x8d;
  if (*Prefix == 0x4c)
    *Prefix = (IsMov || RspAdd) ? 0x49 : 0x4d;
  *RegSlot = (IsMov || RspAdd) ? (0xc0 | Reg) : (0x80 | Reg | (Reg << 3));
  relocateOne(Loc, BufEnd, R_X86_64_TPOFF32, P, SA);
}

// This function applies a TLS relocation with an optimization as described
// in the Ulrich's document. As a result of rewriting instructions at the
// relocation target, relocations immediately follow the TLS relocation (which
// would be applied to rewritten instructions) may have to be skipped.
// This function returns a number of relocations that need to be skipped.
unsigned X86_64TargetInfo::relocateTlsOptimize(uint8_t *Loc, uint8_t *BufEnd,
                                               uint32_t Type, uint64_t P,
                                               uint64_t SA) const {
  switch (Type) {
  case R_X86_64_GOTTPOFF:
    relocateTlsIeToLe(Loc, BufEnd, P, SA);
    return 0;
  case R_X86_64_TLSLD:
    relocateTlsLdToLe(Loc, BufEnd, P, SA);
    // The next relocation should be against __tls_get_addr, so skip it
    return 1;
  case R_X86_64_TLSGD:
    relocateTlsGdToLe(Loc, BufEnd, P, SA);
    // The next relocation should be against __tls_get_addr, so skip it
    return 1;
  case R_X86_64_DTPOFF32:
    relocateOne(Loc, BufEnd, R_X86_64_TPOFF32, P, SA);
    return 0;
  }
  llvm_unreachable("Unknown TLS optimization");
}

void X86_64TargetInfo::relocateOne(uint8_t *Loc, uint8_t *BufEnd, uint32_t Type,
                                   uint64_t P, uint64_t SA) const {
  switch (Type) {
  case R_X86_64_PC32:
  case R_X86_64_GOTPCREL:
  case R_X86_64_PLT32:
  case R_X86_64_TLSLD:
  case R_X86_64_TLSGD:
  case R_X86_64_TPOFF64:
    write32le(Loc, SA - P);
    break;
  case R_X86_64_64:
  case R_X86_64_DTPOFF64:
    write64le(Loc, SA);
    break;
  case R_X86_64_32:
    checkUInt<32>(SA, Type);
    write32le(Loc, SA);
    break;
  case R_X86_64_32S:
    checkInt<32>(SA, Type);
    write32le(Loc, SA);
    break;
  case R_X86_64_DTPOFF32:
    write32le(Loc, SA);
    break;
  case R_X86_64_TPOFF32: {
    uint64_t Val = SA - Out<ELF64LE>::TlsPhdr->p_memsz;
    checkInt<32>(Val, Type);
    write32le(Loc, Val);
    break;
  }
  default:
    error("unrecognized reloc " + Twine(Type));
  }
}

// Relocation masks following the #lo(value), #hi(value), #ha(value),
// #higher(value), #highera(value), #highest(value), and #highesta(value)
// macros defined in section 4.5.1. Relocation Types of the PPC-elf64abi
// document.
static uint16_t applyPPCLo(uint64_t V) { return V; }
static uint16_t applyPPCHi(uint64_t V) { return V >> 16; }
static uint16_t applyPPCHa(uint64_t V) { return (V + 0x8000) >> 16; }
static uint16_t applyPPCHigher(uint64_t V) { return V >> 32; }
static uint16_t applyPPCHighera(uint64_t V) { return (V + 0x8000) >> 32; }
static uint16_t applyPPCHighest(uint64_t V) { return V >> 48; }
static uint16_t applyPPCHighesta(uint64_t V) { return (V + 0x8000) >> 48; }

PPC64TargetInfo::PPC64TargetInfo() {
  PCRelReloc = R_PPC64_REL24;
  GotReloc = R_PPC64_GLOB_DAT;
  GotRefReloc = R_PPC64_REL64;
  RelativeReloc = R_PPC64_RELATIVE;
  PltEntrySize = 32;

  // We need 64K pages (at least under glibc/Linux, the loader won't
  // set different permissions on a finer granularity than that).
  PageSize = 65536;

  // The PPC64 ELF ABI v1 spec, says:
  //
  //   It is normally desirable to put segments with different characteristics
  //   in separate 256 Mbyte portions of the address space, to give the
  //   operating system full paging flexibility in the 64-bit address space.
  //
  // And because the lowest non-zero 256M boundary is 0x10000000, PPC64 linkers
  // use 0x10000000 as the starting address.
  VAStart = 0x10000000;
}

uint64_t getPPC64TocBase() {
  // The TOC consists of sections .got, .toc, .tocbss, .plt in that
  // order. The TOC starts where the first of these sections starts.

  // FIXME: This obviously does not do the right thing when there is no .got
  // section, but there is a .toc or .tocbss section.
  uint64_t TocVA = Out<ELF64BE>::Got->getVA();
  if (!TocVA)
    TocVA = Out<ELF64BE>::Plt->getVA();

  // Per the ppc64-elf-linux ABI, The TOC base is TOC value plus 0x8000
  // thus permitting a full 64 Kbytes segment. Note that the glibc startup
  // code (crt1.o) assumes that you can get from the TOC base to the
  // start of the .toc section with only a single (signed) 16-bit relocation.
  return TocVA + 0x8000;
}

void PPC64TargetInfo::writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const {}
void PPC64TargetInfo::writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                        uint64_t PltEntryAddr) const {}
void PPC64TargetInfo::writePltEntry(uint8_t *Buf, uint64_t GotAddr,
                                    uint64_t GotEntryAddr,
                                    uint64_t PltEntryAddr, int32_t Index,
                                    unsigned RelOff) const {
  uint64_t Off = GotEntryAddr - getPPC64TocBase();

  // FIXME: What we should do, in theory, is get the offset of the function
  // descriptor in the .opd section, and use that as the offset from %r2 (the
  // TOC-base pointer). Instead, we have the GOT-entry offset, and that will
  // be a pointer to the function descriptor in the .opd section. Using
  // this scheme is simpler, but requires an extra indirection per PLT dispatch.

  write32be(Buf,      0xf8410028);                   // std %r2, 40(%r1)
  write32be(Buf + 4,  0x3d620000 | applyPPCHa(Off)); // addis %r11, %r2, X@ha
  write32be(Buf + 8,  0xe98b0000 | applyPPCLo(Off)); // ld %r12, X@l(%r11)
  write32be(Buf + 12, 0xe96c0000);                   // ld %r11,0(%r12)
  write32be(Buf + 16, 0x7d6903a6);                   // mtctr %r11
  write32be(Buf + 20, 0xe84c0008);                   // ld %r2,8(%r12)
  write32be(Buf + 24, 0xe96c0010);                   // ld %r11,16(%r12)
  write32be(Buf + 28, 0x4e800420);                   // bctr
}

bool PPC64TargetInfo::relocNeedsGot(uint32_t Type, const SymbolBody &S) const {
  if (relocNeedsPlt(Type, S))
    return true;

  switch (Type) {
  default: return false;
  case R_PPC64_GOT16:
  case R_PPC64_GOT16_LO:
  case R_PPC64_GOT16_HI:
  case R_PPC64_GOT16_HA:
  case R_PPC64_GOT16_DS:
  case R_PPC64_GOT16_LO_DS:
    return true;
  }
}

bool PPC64TargetInfo::relocNeedsPlt(uint32_t Type, const SymbolBody &S) const {
  // These are function calls that need to be redirected through a PLT stub.
  return Type == R_PPC64_REL24 && canBePreempted(&S, false);
}

bool PPC64TargetInfo::isRelRelative(uint32_t Type) const {
  switch (Type) {
  default:
    return true;
  case R_PPC64_TOC:
  case R_PPC64_ADDR64:
    return false;
  }
}

void PPC64TargetInfo::relocateOne(uint8_t *Loc, uint8_t *BufEnd, uint32_t Type,
                                  uint64_t P, uint64_t SA) const {
  uint64_t TB = getPPC64TocBase();

  // For a TOC-relative relocation, adjust the addend and proceed in terms of
  // the corresponding ADDR16 relocation type.
  switch (Type) {
  case R_PPC64_TOC16:       Type = R_PPC64_ADDR16;       SA -= TB; break;
  case R_PPC64_TOC16_DS:    Type = R_PPC64_ADDR16_DS;    SA -= TB; break;
  case R_PPC64_TOC16_LO:    Type = R_PPC64_ADDR16_LO;    SA -= TB; break;
  case R_PPC64_TOC16_LO_DS: Type = R_PPC64_ADDR16_LO_DS; SA -= TB; break;
  case R_PPC64_TOC16_HI:    Type = R_PPC64_ADDR16_HI;    SA -= TB; break;
  case R_PPC64_TOC16_HA:    Type = R_PPC64_ADDR16_HA;    SA -= TB; break;
  default: break;
  }

  switch (Type) {
  case R_PPC64_ADDR16:
    checkInt<16>(SA, Type);
    write16be(Loc, SA);
    break;
  case R_PPC64_ADDR16_DS:
    checkInt<16>(SA, Type);
    write16be(Loc, (read16be(Loc) & 3) | (SA & ~3));
    break;
  case R_PPC64_ADDR16_LO:
    write16be(Loc, applyPPCLo(SA));
    break;
  case R_PPC64_ADDR16_LO_DS:
    write16be(Loc, (read16be(Loc) & 3) | (applyPPCLo(SA) & ~3));
    break;
  case R_PPC64_ADDR16_HI:
    write16be(Loc, applyPPCHi(SA));
    break;
  case R_PPC64_ADDR16_HA:
    write16be(Loc, applyPPCHa(SA));
    break;
  case R_PPC64_ADDR16_HIGHER:
    write16be(Loc, applyPPCHigher(SA));
    break;
  case R_PPC64_ADDR16_HIGHERA:
    write16be(Loc, applyPPCHighera(SA));
    break;
  case R_PPC64_ADDR16_HIGHEST:
    write16be(Loc, applyPPCHighest(SA));
    break;
  case R_PPC64_ADDR16_HIGHESTA:
    write16be(Loc, applyPPCHighesta(SA));
    break;
  case R_PPC64_ADDR14: {
    checkAlignment<4>(SA, Type);
    // Preserve the AA/LK bits in the branch instruction
    uint8_t AALK = Loc[3];
    write16be(Loc + 2, (AALK & 3) | (SA & 0xfffc));
    break;
  }
  case R_PPC64_REL16_LO:
    write16be(Loc, applyPPCLo(SA - P));
    break;
  case R_PPC64_REL16_HI:
    write16be(Loc, applyPPCHi(SA - P));
    break;
  case R_PPC64_REL16_HA:
    write16be(Loc, applyPPCHa(SA - P));
    break;
  case R_PPC64_ADDR32:
    checkInt<32>(SA, Type);
    write32be(Loc, SA);
    break;
  case R_PPC64_REL24: {
    // If we have an undefined weak symbol, we might get here with a symbol
    // address of zero. That could overflow, but the code must be unreachable,
    // so don't bother doing anything at all.
    if (!SA)
      break;

    uint64_t PltStart = Out<ELF64BE>::Plt->getVA();
    uint64_t PltEnd = PltStart + Out<ELF64BE>::Plt->getSize();
    bool InPlt = PltStart <= SA && SA < PltEnd;

    if (!InPlt && Out<ELF64BE>::Opd) {
      // If this is a local call, and we currently have the address of a
      // function-descriptor, get the underlying code address instead.
      uint64_t OpdStart = Out<ELF64BE>::Opd->getVA();
      uint64_t OpdEnd = OpdStart + Out<ELF64BE>::Opd->getSize();
      bool InOpd = OpdStart <= SA && SA < OpdEnd;

      if (InOpd)
        SA = read64be(&Out<ELF64BE>::OpdBuf[SA - OpdStart]);
    }

    uint32_t Mask = 0x03FFFFFC;
    checkInt<24>(SA - P, Type);
    write32be(Loc, (read32be(Loc) & ~Mask) | ((SA - P) & Mask));

    uint32_t Nop = 0x60000000;
    if (InPlt && Loc + 8 <= BufEnd && read32be(Loc + 4) == Nop)
      write32be(Loc + 4, 0xe8410028); // ld %r2, 40(%r1)
    break;
  }
  case R_PPC64_REL32:
    checkInt<32>(SA - P, Type);
    write32be(Loc, SA - P);
    break;
  case R_PPC64_REL64:
    write64be(Loc, SA - P);
    break;
  case R_PPC64_ADDR64:
  case R_PPC64_TOC:
    write64be(Loc, SA);
    break;
  default:
    error("unrecognized reloc " + Twine(Type));
  }
}

AArch64TargetInfo::AArch64TargetInfo() {
  GotReloc = R_AARCH64_GLOB_DAT;
  PltReloc = R_AARCH64_JUMP_SLOT;
  LazyRelocations = true;
  PltEntrySize = 16;
  PltZeroEntrySize = 32;
}

unsigned AArch64TargetInfo::getGotRefReloc(unsigned Type) const { return Type; }

unsigned AArch64TargetInfo::getPltRefReloc(unsigned Type) const { return Type; }

void AArch64TargetInfo::writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const {
  write64le(Buf, Out<ELF64LE>::Plt->getVA());
}

void AArch64TargetInfo::writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                          uint64_t PltEntryAddr) const {
  const uint8_t PltData[] = {
      0xf0, 0x7b, 0xbf, 0xa9, // stp	x16, x30, [sp,#-16]!
      0x10, 0x00, 0x00, 0x90, // adrp	x16, Page(&(.plt.got[2]))
      0x11, 0x02, 0x40, 0xf9, // ldr	x17, [x16, Offset(&(.plt.got[2]))]
      0x10, 0x02, 0x00, 0x91, // add	x16, x16, Offset(&(.plt.got[2]))
      0x20, 0x02, 0x1f, 0xd6, // br	x17
      0x1f, 0x20, 0x03, 0xd5, // nop
      0x1f, 0x20, 0x03, 0xd5, // nop
      0x1f, 0x20, 0x03, 0xd5  // nop
  };
  memcpy(Buf, PltData, sizeof(PltData));

  relocateOne(Buf + 4, Buf + 8, R_AARCH64_ADR_PREL_PG_HI21, PltEntryAddr + 4,
              GotEntryAddr + 16);
  relocateOne(Buf + 8, Buf + 12, R_AARCH64_LDST64_ABS_LO12_NC, PltEntryAddr + 8,
              GotEntryAddr + 16);
  relocateOne(Buf + 12, Buf + 16, R_AARCH64_ADD_ABS_LO12_NC, PltEntryAddr + 12,
              GotEntryAddr + 16);
}

void AArch64TargetInfo::writePltEntry(uint8_t *Buf, uint64_t GotAddr,
                                      uint64_t GotEntryAddr,
                                      uint64_t PltEntryAddr, int32_t Index,
                                      unsigned RelOff) const {
  const uint8_t Inst[] = {
      0x10, 0x00, 0x00, 0x90, // adrp x16, Page(&(.plt.got[n]))
      0x11, 0x02, 0x40, 0xf9, // ldr  x17, [x16, Offset(&(.plt.got[n]))]
      0x10, 0x02, 0x00, 0x91, // add  x16, x16, Offset(&(.plt.got[n]))
      0x20, 0x02, 0x1f, 0xd6  // br   x17
  };
  memcpy(Buf, Inst, sizeof(Inst));

  relocateOne(Buf, Buf + 4, R_AARCH64_ADR_PREL_PG_HI21, PltEntryAddr,
              GotEntryAddr);
  relocateOne(Buf + 4, Buf + 8, R_AARCH64_LDST64_ABS_LO12_NC, PltEntryAddr + 4,
              GotEntryAddr);
  relocateOne(Buf + 8, Buf + 12, R_AARCH64_ADD_ABS_LO12_NC, PltEntryAddr + 8,
              GotEntryAddr);
}

bool AArch64TargetInfo::relocNeedsGot(uint32_t Type,
                                      const SymbolBody &S) const {
  return Type == R_AARCH64_ADR_GOT_PAGE || Type == R_AARCH64_LD64_GOT_LO12_NC ||
         relocNeedsPlt(Type, S);
}

bool AArch64TargetInfo::relocNeedsPlt(uint32_t Type,
                                      const SymbolBody &S) const {
  switch (Type) {
  default:
    return false;
  case R_AARCH64_JUMP26:
  case R_AARCH64_CALL26:
    return canBePreempted(&S, true);
  }
}

static void updateAArch64Adr(uint8_t *L, uint64_t Imm) {
  uint32_t ImmLo = (Imm & 0x3) << 29;
  uint32_t ImmHi = ((Imm & 0x1FFFFC) >> 2) << 5;
  uint64_t Mask = (0x3 << 29) | (0x7FFFF << 5);
  write32le(L, (read32le(L) & ~Mask) | ImmLo | ImmHi);
}

// Page(Expr) is the page address of the expression Expr, defined
// as (Expr & ~0xFFF). (This applies even if the machine page size
// supported by the platform has a different value.)
static uint64_t getAArch64Page(uint64_t Expr) {
  return Expr & (~static_cast<uint64_t>(0xFFF));
}

void AArch64TargetInfo::relocateOne(uint8_t *Loc, uint8_t *BufEnd,
                                    uint32_t Type, uint64_t P,
                                    uint64_t SA) const {
  switch (Type) {
  case R_AARCH64_ABS16:
    checkIntUInt<16>(SA, Type);
    write16le(Loc, SA);
    break;
  case R_AARCH64_ABS32:
    checkIntUInt<32>(SA, Type);
    write32le(Loc, SA);
    break;
  case R_AARCH64_ABS64:
    // No overflow check needed.
    write64le(Loc, SA);
    break;
  case R_AARCH64_ADD_ABS_LO12_NC:
    // No overflow check needed.
    // This relocation stores 12 bits and there's no instruction
    // to do it. Instead, we do a 32 bits store of the value
    // of r_addend bitwise-or'ed Loc. This assumes that the addend
    // bits in Loc are zero.
    or32le(Loc, (SA & 0xFFF) << 10);
    break;
  case R_AARCH64_ADR_PREL_LO21: {
    uint64_t X = SA - P;
    checkInt<21>(X, Type);
    updateAArch64Adr(Loc, X & 0x1FFFFF);
    break;
  }
  case R_AARCH64_ADR_GOT_PAGE:
  case R_AARCH64_ADR_PREL_PG_HI21: {
    uint64_t X = getAArch64Page(SA) - getAArch64Page(P);
    checkInt<33>(X, Type);
    updateAArch64Adr(Loc, (X >> 12) & 0x1FFFFF); // X[32:12]
    break;
  }
  case R_AARCH64_JUMP26:
  case R_AARCH64_CALL26: {
    uint64_t X = SA - P;
    checkInt<28>(X, Type);
    or32le(Loc, (X & 0x0FFFFFFC) >> 2);
    break;
  }
  case R_AARCH64_LDST32_ABS_LO12_NC:
    // No overflow check needed.
    or32le(Loc, (SA & 0xFFC) << 8);
    break;
  case R_AARCH64_LD64_GOT_LO12_NC:
    checkAlignment<8>(SA, Type);
    // No overflow check needed.
    or32le(Loc, (SA & 0xFF8) << 7);
    break;
  case R_AARCH64_LDST64_ABS_LO12_NC:
    // No overflow check needed.
    or32le(Loc, (SA & 0xFF8) << 7);
    break;
  case R_AARCH64_LDST8_ABS_LO12_NC:
    // No overflow check needed.
    or32le(Loc, (SA & 0xFFF) << 10);
    break;
  case R_AARCH64_PREL16:
    checkIntUInt<16>(SA - P, Type);
    write16le(Loc, SA - P);
    break;
  case R_AARCH64_PREL32:
    checkIntUInt<32>(SA - P, Type);
    write32le(Loc, SA - P);
    break;
  case R_AARCH64_PREL64:
    // No overflow check needed.
    write64le(Loc, SA - P);
    break;
  default:
    error("unrecognized reloc " + Twine(Type));
  }
}

template <class ELFT> MipsTargetInfo<ELFT>::MipsTargetInfo() {
  PageSize = 65536;
  GotRefReloc = R_MIPS_GOT16;
  GotHeaderEntriesNum = 2;
}

template <class ELFT>
unsigned MipsTargetInfo<ELFT>::getGotRefReloc(unsigned Type) const {
  return Type;
}

template <class ELFT>
void MipsTargetInfo<ELFT>::writeGotHeaderEntries(uint8_t *Buf) const {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Off Elf_Off;
  auto *P = reinterpret_cast<Elf_Off *>(Buf);
  // Module pointer
  P[1] = ELFT::Is64Bits ? 0x8000000000000000 : 0x80000000;
}

template <class ELFT>
void MipsTargetInfo<ELFT>::writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const {}
template <class ELFT>
void MipsTargetInfo<ELFT>::writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                       uint64_t PltEntryAddr) const {}
template <class ELFT>
void MipsTargetInfo<ELFT>::writePltEntry(uint8_t *Buf, uint64_t GotAddr,
                                         uint64_t GotEntryAddr,
                                         uint64_t PltEntryAddr, int32_t Index,
                                         unsigned RelOff) const {}

template <class ELFT>
bool MipsTargetInfo<ELFT>::relocNeedsGot(uint32_t Type,
                                         const SymbolBody &S) const {
  return Type == R_MIPS_GOT16 || Type == R_MIPS_CALL16;
}

template <class ELFT>
bool MipsTargetInfo<ELFT>::relocNeedsPlt(uint32_t Type,
                                         const SymbolBody &S) const {
  return false;
}

template <class ELFT>
void MipsTargetInfo<ELFT>::relocateOne(uint8_t *Loc, uint8_t *BufEnd,
                                       uint32_t Type, uint64_t P,
                                       uint64_t SA) const {
  const endianness E = ELFT::TargetEndianness;
  switch (Type) {
  case R_MIPS_32:
    add32<E>(Loc, SA);
    break;
  case R_MIPS_CALL16:
  case R_MIPS_GOT16: {
    int64_t V = SA - getMipsGpAddr<ELFT>();
    if (Type == R_MIPS_GOT16)
      checkInt<16>(V, Type);
    write32<E>(Loc, (read32<E>(Loc) & 0xffff0000) | (V & 0xffff));
    break;
  }
  default:
    error("unrecognized reloc " + Twine(Type));
  }
}

template <class ELFT>
typename llvm::object::ELFFile<ELFT>::uintX_t getMipsGpAddr() {
  const unsigned GPOffset = 0x7ff0;
  return Out<ELFT>::Got->getVA() ? (Out<ELFT>::Got->getVA() + GPOffset) : 0;
}

template uint32_t getMipsGpAddr<ELF32LE>();
template uint32_t getMipsGpAddr<ELF32BE>();
template uint64_t getMipsGpAddr<ELF64LE>();
template uint64_t getMipsGpAddr<ELF64BE>();
}
}
