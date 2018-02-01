//===- X86.cpp ------------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "InputFiles.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace {
class X86 : public TargetInfo {
public:
  X86();
  RelExpr getRelExpr(uint32_t Type, const SymbolBody &S,
                     const uint8_t *Loc) const override;
  int64_t getImplicitAddend(const uint8_t *Buf, uint32_t Type) const override;
  void writeGotPltHeader(uint8_t *Buf) const override;
  uint32_t getDynRel(uint32_t Type) const override;
  void writeGotPlt(uint8_t *Buf, const SymbolBody &S) const override;
  void writeIgotPlt(uint8_t *Buf, const SymbolBody &S) const override;
  void writePltHeader(uint8_t *Buf) const override;
  void writePlt(uint8_t *Buf, uint64_t GotPltEntryAddr, uint64_t PltEntryAddr,
                int32_t Index, unsigned RelOff) const override;
  void relocateOne(uint8_t *Loc, uint32_t Type, uint64_t Val) const override;

  RelExpr adjustRelaxExpr(uint32_t Type, const uint8_t *Data,
                          RelExpr Expr) const override;
  void relaxTlsGdToIe(uint8_t *Loc, uint32_t Type, uint64_t Val) const override;
  void relaxTlsGdToLe(uint8_t *Loc, uint32_t Type, uint64_t Val) const override;
  void relaxTlsIeToLe(uint8_t *Loc, uint32_t Type, uint64_t Val) const override;
  void relaxTlsLdToLe(uint8_t *Loc, uint32_t Type, uint64_t Val) const override;
};
} // namespace

X86::X86() {
  GotBaseSymOff = -1;
  CopyRel = R_386_COPY;
  GotRel = R_386_GLOB_DAT;
  PltRel = R_386_JUMP_SLOT;
  IRelativeRel = R_386_IRELATIVE;
  RelativeRel = R_386_RELATIVE;
  TlsGotRel = R_386_TLS_TPOFF;
  TlsModuleIndexRel = R_386_TLS_DTPMOD32;
  TlsOffsetRel = R_386_TLS_DTPOFF32;
  GotEntrySize = 4;
  GotPltEntrySize = 4;
  PltEntrySize = 16;
  PltHeaderSize = 16;
  TlsGdRelaxSkip = 2;
  TrapInstr = 0xcccccccc; // 0xcc = INT3
}

RelExpr X86::getRelExpr(uint32_t Type, const SymbolBody &S,
                        const uint8_t *Loc) const {
  switch (Type) {
  case R_386_8:
  case R_386_16:
  case R_386_32:
  case R_386_TLS_LDO_32:
    return R_ABS;
  case R_386_TLS_GD:
    return R_TLSGD;
  case R_386_TLS_LDM:
    return R_TLSLD;
  case R_386_PLT32:
    return R_PLT_PC;
  case R_386_PC8:
  case R_386_PC16:
  case R_386_PC32:
    return R_PC;
  case R_386_GOTPC:
    return R_GOTONLY_PC_FROM_END;
  case R_386_TLS_IE:
    return R_GOT;
  case R_386_GOT32:
  case R_386_GOT32X:
    // These relocations can be calculated in two different ways.
    // Usual calculation is G + A - GOT what means an offset in GOT table
    // (R_GOT_FROM_END). When instruction pointed by relocation has no base
    // register, then relocations can be used when PIC code is disabled. In that
    // case calculation is G + A, it resolves to an address of entry in GOT
    // (R_GOT) and not an offset.
    //
    // To check that instruction has no base register we scan ModR/M byte.
    // See "Table 2-2. 32-Bit Addressing Forms with the ModR/M Byte"
    // (http://www.intel.com/content/dam/www/public/us/en/documents/manuals/
    //  64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf)
    if ((Loc[-1] & 0xc7) != 0x5)
      return R_GOT_FROM_END;
    if (Config->Pic)
      error(toString(S.File) + ": relocation " + toString(Type) + " against '" +
            S.getName() +
            "' without base register can not be used when PIC enabled");
    return R_GOT;
  case R_386_TLS_GOTIE:
    return R_GOT_FROM_END;
  case R_386_GOTOFF:
    return R_GOTREL_FROM_END;
  case R_386_TLS_LE:
    return R_TLS;
  case R_386_TLS_LE_32:
    return R_NEG_TLS;
  case R_386_NONE:
    return R_NONE;
  default:
    error(toString(S.File) + ": unknown relocation type: " + toString(Type));
    return R_HINT;
  }
}

RelExpr X86::adjustRelaxExpr(uint32_t Type, const uint8_t *Data,
                             RelExpr Expr) const {
  switch (Expr) {
  default:
    return Expr;
  case R_RELAX_TLS_GD_TO_IE:
    return R_RELAX_TLS_GD_TO_IE_END;
  case R_RELAX_TLS_GD_TO_LE:
    return R_RELAX_TLS_GD_TO_LE_NEG;
  }
}

void X86::writeGotPltHeader(uint8_t *Buf) const {
  write32le(Buf, InX::Dynamic->getVA());
}

void X86::writeGotPlt(uint8_t *Buf, const SymbolBody &S) const {
  // Entries in .got.plt initially points back to the corresponding
  // PLT entries with a fixed offset to skip the first instruction.
  write32le(Buf, S.getPltVA() + 6);
}

void X86::writeIgotPlt(uint8_t *Buf, const SymbolBody &S) const {
  // An x86 entry is the address of the ifunc resolver function.
  write32le(Buf, S.getVA());
}

uint32_t X86::getDynRel(uint32_t Type) const {
  if (Type == R_386_TLS_LE)
    return R_386_TLS_TPOFF;
  if (Type == R_386_TLS_LE_32)
    return R_386_TLS_TPOFF32;
  return Type;
}

void X86::writePltHeader(uint8_t *Buf) const {
  if (Config->Pic) {
    const uint8_t V[] = {
        0xff, 0xb3, 0x04, 0x00, 0x00, 0x00, // pushl GOTPLT+4(%ebx)
        0xff, 0xa3, 0x08, 0x00, 0x00, 0x00, // jmp *GOTPLT+8(%ebx)
        0x90, 0x90, 0x90, 0x90              // nop
    };
    memcpy(Buf, V, sizeof(V));

    uint32_t Ebx = InX::Got->getVA() + InX::Got->getSize();
    uint32_t GotPlt = InX::GotPlt->getVA() - Ebx;
    write32le(Buf + 2, GotPlt + 4);
    write32le(Buf + 8, GotPlt + 8);
    return;
  }

  const uint8_t PltData[] = {
      0xff, 0x35, 0x00, 0x00, 0x00, 0x00, // pushl (GOTPLT+4)
      0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // jmp *(GOTPLT+8)
      0x90, 0x90, 0x90, 0x90              // nop
  };
  memcpy(Buf, PltData, sizeof(PltData));
  uint32_t GotPlt = InX::GotPlt->getVA();
  write32le(Buf + 2, GotPlt + 4);
  write32le(Buf + 8, GotPlt + 8);
}

void X86::writePlt(uint8_t *Buf, uint64_t GotPltEntryAddr,
                   uint64_t PltEntryAddr, int32_t Index,
                   unsigned RelOff) const {
  const uint8_t Inst[] = {
      0xff, 0x00, 0x00, 0x00, 0x00, 0x00, // jmp *foo_in_GOT|*foo@GOT(%ebx)
      0x68, 0x00, 0x00, 0x00, 0x00,       // pushl $reloc_offset
      0xe9, 0x00, 0x00, 0x00, 0x00        // jmp .PLT0@PC
  };
  memcpy(Buf, Inst, sizeof(Inst));

  if (Config->Pic) {
    // jmp *foo@GOT(%ebx)
    uint32_t Ebx = InX::Got->getVA() + InX::Got->getSize();
    Buf[1] = 0xa3;
    write32le(Buf + 2, GotPltEntryAddr - Ebx);
  } else {
    // jmp *foo_in_GOT
    Buf[1] = 0x25;
    write32le(Buf + 2, GotPltEntryAddr);
  }

  write32le(Buf + 7, RelOff);
  write32le(Buf + 12, -Index * PltEntrySize - PltHeaderSize - 16);
}

int64_t X86::getImplicitAddend(const uint8_t *Buf, uint32_t Type) const {
  switch (Type) {
  default:
    return 0;
  case R_386_8:
  case R_386_PC8:
    return SignExtend64<8>(*Buf);
  case R_386_16:
  case R_386_PC16:
    return SignExtend64<16>(read16le(Buf));
  case R_386_32:
  case R_386_GOT32:
  case R_386_GOT32X:
  case R_386_GOTOFF:
  case R_386_GOTPC:
  case R_386_PC32:
  case R_386_PLT32:
  case R_386_TLS_LDO_32:
  case R_386_TLS_LE:
    return SignExtend64<32>(read32le(Buf));
  }
}

void X86::relocateOne(uint8_t *Loc, uint32_t Type, uint64_t Val) const {
  // R_386_{PC,}{8,16} are not part of the i386 psABI, but they are
  // being used for some 16-bit programs such as boot loaders, so
  // we want to support them.
  switch (Type) {
  case R_386_8:
    checkUInt<8>(Loc, Val, Type);
    *Loc = Val;
    break;
  case R_386_PC8:
    checkInt<8>(Loc, Val, Type);
    *Loc = Val;
    break;
  case R_386_16:
    checkUInt<16>(Loc, Val, Type);
    write16le(Loc, Val);
    break;
  case R_386_PC16:
    // R_386_PC16 is normally used with 16 bit code. In that situation
    // the PC is 16 bits, just like the addend. This means that it can
    // point from any 16 bit address to any other if the possibility
    // of wrapping is included.
    // The only restriction we have to check then is that the destination
    // address fits in 16 bits. That is impossible to do here. The problem is
    // that we are passed the final value, which already had the
    // current location subtracted from it.
    // We just check that Val fits in 17 bits. This misses some cases, but
    // should have no false positives.
    checkInt<17>(Loc, Val, Type);
    write16le(Loc, Val);
    break;
  default:
    checkInt<32>(Loc, Val, Type);
    write32le(Loc, Val);
  }
}

void X86::relaxTlsGdToLe(uint8_t *Loc, uint32_t Type, uint64_t Val) const {
  // Convert
  //   leal x@tlsgd(, %ebx, 1),
  //   call __tls_get_addr@plt
  // to
  //   movl %gs:0,%eax
  //   subl $x@ntpoff,%eax
  const uint8_t Inst[] = {
      0x65, 0xa1, 0x00, 0x00, 0x00, 0x00, // movl %gs:0, %eax
      0x81, 0xe8, 0x00, 0x00, 0x00, 0x00  // subl 0(%ebx), %eax
  };
  memcpy(Loc - 3, Inst, sizeof(Inst));
  write32le(Loc + 5, Val);
}

void X86::relaxTlsGdToIe(uint8_t *Loc, uint32_t Type, uint64_t Val) const {
  // Convert
  //   leal x@tlsgd(, %ebx, 1),
  //   call __tls_get_addr@plt
  // to
  //   movl %gs:0, %eax
  //   addl x@gotntpoff(%ebx), %eax
  const uint8_t Inst[] = {
      0x65, 0xa1, 0x00, 0x00, 0x00, 0x00, // movl %gs:0, %eax
      0x03, 0x83, 0x00, 0x00, 0x00, 0x00  // addl 0(%ebx), %eax
  };
  memcpy(Loc - 3, Inst, sizeof(Inst));
  write32le(Loc + 5, Val);
}

// In some conditions, relocations can be optimized to avoid using GOT.
// This function does that for Initial Exec to Local Exec case.
void X86::relaxTlsIeToLe(uint8_t *Loc, uint32_t Type, uint64_t Val) const {
  // Ulrich's document section 6.2 says that @gotntpoff can
  // be used with MOVL or ADDL instructions.
  // @indntpoff is similar to @gotntpoff, but for use in
  // position dependent code.
  uint8_t Reg = (Loc[-1] >> 3) & 7;

  if (Type == R_386_TLS_IE) {
    if (Loc[-1] == 0xa1) {
      // "movl foo@indntpoff,%eax" -> "movl $foo,%eax"
      // This case is different from the generic case below because
      // this is a 5 byte instruction while below is 6 bytes.
      Loc[-1] = 0xb8;
    } else if (Loc[-2] == 0x8b) {
      // "movl foo@indntpoff,%reg" -> "movl $foo,%reg"
      Loc[-2] = 0xc7;
      Loc[-1] = 0xc0 | Reg;
    } else {
      // "addl foo@indntpoff,%reg" -> "addl $foo,%reg"
      Loc[-2] = 0x81;
      Loc[-1] = 0xc0 | Reg;
    }
  } else {
    assert(Type == R_386_TLS_GOTIE);
    if (Loc[-2] == 0x8b) {
      // "movl foo@gottpoff(%rip),%reg" -> "movl $foo,%reg"
      Loc[-2] = 0xc7;
      Loc[-1] = 0xc0 | Reg;
    } else {
      // "addl foo@gotntpoff(%rip),%reg" -> "leal foo(%reg),%reg"
      Loc[-2] = 0x8d;
      Loc[-1] = 0x80 | (Reg << 3) | Reg;
    }
  }
  write32le(Loc, Val);
}

void X86::relaxTlsLdToLe(uint8_t *Loc, uint32_t Type, uint64_t Val) const {
  if (Type == R_386_TLS_LDO_32) {
    write32le(Loc, Val);
    return;
  }

  // Convert
  //   leal foo(%reg),%eax
  //   call ___tls_get_addr
  // to
  //   movl %gs:0,%eax
  //   nop
  //   leal 0(%esi,1),%esi
  const uint8_t Inst[] = {
      0x65, 0xa1, 0x00, 0x00, 0x00, 0x00, // movl %gs:0,%eax
      0x90,                               // nop
      0x8d, 0x74, 0x26, 0x00              // leal 0(%esi,1),%esi
  };
  memcpy(Loc - 2, Inst, sizeof(Inst));
}

namespace {
class RetpolinePic : public X86 {
public:
  RetpolinePic();
  void writeGotPlt(uint8_t *Buf, const SymbolBody &S) const override;
  void writePltHeader(uint8_t *Buf) const override;
  void writePlt(uint8_t *Buf, uint64_t GotPltEntryAddr, uint64_t PltEntryAddr,
                int32_t Index, unsigned RelOff) const override;
};

class RetpolineNoPic : public X86 {
public:
  RetpolineNoPic();
  void writeGotPlt(uint8_t *Buf, const SymbolBody &S) const override;
  void writePltHeader(uint8_t *Buf) const override;
  void writePlt(uint8_t *Buf, uint64_t GotPltEntryAddr, uint64_t PltEntryAddr,
                int32_t Index, unsigned RelOff) const override;
};
} // namespace

RetpolinePic::RetpolinePic() {
  PltHeaderSize = 48;
  PltEntrySize = 32;
}

void RetpolinePic::writeGotPlt(uint8_t *Buf, const SymbolBody &S) const {
  write32le(Buf, S.getPltVA() + 17);
}

void RetpolinePic::writePltHeader(uint8_t *Buf) const {
  const uint8_t Insn[] = {
      0xff, 0xb3, 0,    0,    0,    0,          // 0:    pushl GOTPLT+4(%ebx)
      0x50,                                     // 6:    pushl %eax
      0x8b, 0x83, 0,    0,    0,    0,          // 7:    mov GOTPLT+8(%ebx), %eax
      0xe8, 0x0e, 0x00, 0x00, 0x00,             // d:    call next
      0xf3, 0x90,                               // 12: loop: pause
      0x0f, 0xae, 0xe8,                         // 14:   lfence
      0xeb, 0xf9,                               // 17:   jmp loop
      0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, // 19:   int3; .align 16
      0x89, 0x0c, 0x24,                         // 20: next: mov %ecx, (%esp)
      0x8b, 0x4c, 0x24, 0x04,                   // 23:   mov 0x4(%esp), %ecx
      0x89, 0x44, 0x24, 0x04,                   // 27:   mov %eax ,0x4(%esp)
      0x89, 0xc8,                               // 2b:   mov %ecx, %eax
      0x59,                                     // 2d:   pop %ecx
      0xc3,                                     // 2e:   ret
      0xcc,                                     // 2f:   int3
  };
  memcpy(Buf, Insn, sizeof(Insn));
  assert(sizeof(Insn) == TargetInfo::PltHeaderSize);

  uint32_t Ebx = InX::Got->getVA() + InX::Got->getSize();
  uint32_t GotPlt = InX::GotPlt->getVA() - Ebx;
  write32le(Buf + 2, GotPlt + 4);
  write32le(Buf + 9, GotPlt + 8);
}

void RetpolinePic::writePlt(uint8_t *Buf, uint64_t GotPltEntryAddr,
                            uint64_t PltEntryAddr, int32_t Index,
                            unsigned RelOff) const {
  const uint8_t Insn[] = {
      0x50,                   // pushl %eax
      0x8b, 0x83, 0, 0, 0, 0, // mov foo@GOT(%ebx), %eax
      0xe8, 0,    0, 0, 0,    // call plt+0x20
      0xe9, 0,    0, 0, 0,    // jmp plt+0x12
      0x68, 0,    0, 0, 0,    // pushl $reloc_offset
      0xe9, 0,    0, 0, 0,    // jmp plt+0
      0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
  };
  memcpy(Buf, Insn, sizeof(Insn));
  assert(sizeof(Insn) == TargetInfo::PltEntrySize);

  uint32_t Ebx = InX::Got->getVA() + InX::Got->getSize();
  write32le(Buf + 3, GotPltEntryAddr - Ebx);
  write32le(Buf + 8, -Index * PltEntrySize - PltHeaderSize - 12 + 32);
  write32le(Buf + 13, -Index * PltEntrySize - PltHeaderSize - 17 + 18);
  write32le(Buf + 18, RelOff);
  write32le(Buf + 23, -Index * PltEntrySize - PltHeaderSize - 27);
}

RetpolineNoPic::RetpolineNoPic() {
  PltHeaderSize = 48;
  PltEntrySize = 32;
}

void RetpolineNoPic::writeGotPlt(uint8_t *Buf, const SymbolBody &S) const {
  write32le(Buf, S.getPltVA() + 16);
}

void RetpolineNoPic::writePltHeader(uint8_t *Buf) const {
  const uint8_t PltData[] = {
      0xff, 0x35, 0,    0,    0,    0, // 0:    pushl GOTPLT+4
      0x50,                            // 6:    pushl %eax
      0xa1, 0,    0,    0,    0,       // 7:    mov GOTPLT+8, %eax
      0xe8, 0x0f, 0x00, 0x00, 0x00,    // c:    call next
      0xf3, 0x90,                      // 11: loop: pause
      0x0f, 0xae, 0xe8,                // 13:   lfence
      0xeb, 0xf9,                      // 16:   jmp loop
      0xcc, 0xcc, 0xcc, 0xcc, 0xcc,    // 18:   int3
      0xcc, 0xcc, 0xcc,                // 1f:   int3; .align 16
      0x89, 0x0c, 0x24,                // 20: next: mov %ecx, (%esp)
      0x8b, 0x4c, 0x24, 0x04,          // 23:   mov 0x4(%esp), %ecx
      0x89, 0x44, 0x24, 0x04,          // 27:   mov %eax ,0x4(%esp)
      0x89, 0xc8,                      // 2b:   mov %ecx, %eax
      0x59,                            // 2d:   pop %ecx
      0xc3,                            // 2e:   ret
      0xcc,                            // 2f:   int3
  };
  memcpy(Buf, PltData, sizeof(PltData));
  assert(sizeof(PltData) == TargetInfo::PltHeaderSize);

  uint32_t GotPlt = InX::GotPlt->getVA();
  write32le(Buf + 2, GotPlt + 4);
  write32le(Buf + 8, GotPlt + 8);
}

void RetpolineNoPic::writePlt(uint8_t *Buf, uint64_t GotPltEntryAddr,
                              uint64_t PltEntryAddr, int32_t Index,
                              unsigned RelOff) const {
  const uint8_t Insn[] = {
      0x50,             // 0:  pushl %eax
      0xa1, 0, 0, 0, 0, // 1:  mov foo_in_GOT, %eax
      0xe8, 0, 0, 0, 0, // 6:  call plt+0x20
      0xe9, 0, 0, 0, 0, // b:  jmp plt+0x11
      0x68, 0, 0, 0, 0, // 10: pushl $reloc_offset
      0xe9, 0, 0, 0, 0, // 15: jmp plt+0
      0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
  };
  memcpy(Buf, Insn, sizeof(Insn));
  assert(sizeof(Insn) == TargetInfo::PltEntrySize);

  write32le(Buf + 2, GotPltEntryAddr);
  write32le(Buf + 7, -Index * PltEntrySize - PltHeaderSize - 11 + 32);
  write32le(Buf + 12, -Index * PltEntrySize - PltHeaderSize - 16 + 17);
  write32le(Buf + 17, RelOff);
  write32le(Buf + 22, -Index * PltEntrySize - PltHeaderSize - 26);
}

TargetInfo *elf::getX86TargetInfo() {
  if (Config->ZRetpolineplt) {
    if (Config->Pic) {
      static RetpolinePic T;
      return &T;
    }
    static RetpolineNoPic T;
    return &T;
  }

  static X86 T;
  return &T;
}
