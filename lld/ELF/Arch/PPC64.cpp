//===- PPC64.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "OutputSections.h"
#include "RelocScan.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "Thunks.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

constexpr uint64_t ppc64TocOffset = 0x8000;
constexpr uint64_t dynamicThreadPointerOffset = 0x8000;

namespace {
// The instruction encoding of bits 21-30 from the ISA for the Xform and Dform
// instructions that can be used as part of the initial exec TLS sequence.
enum XFormOpcd {
  LBZX = 87,
  LHZX = 279,
  LWZX = 23,
  LDX = 21,
  STBX = 215,
  STHX = 407,
  STWX = 151,
  STDX = 149,
  LHAX = 343,
  LWAX = 341,
  LFSX = 535,
  LFDX = 599,
  STFSX = 663,
  STFDX = 727,
  ADD = 266,
};

enum DFormOpcd {
  LBZ = 34,
  LBZU = 35,
  LHZ = 40,
  LHZU = 41,
  LHAU = 43,
  LWZ = 32,
  LWZU = 33,
  LFSU = 49,
  LFDU = 51,
  STB = 38,
  STBU = 39,
  STH = 44,
  STHU = 45,
  STW = 36,
  STWU = 37,
  STFSU = 53,
  STFDU = 55,
  LHA = 42,
  LFS = 48,
  LFD = 50,
  STFS = 52,
  STFD = 54,
  ADDI = 14
};

enum DSFormOpcd {
  LD = 58,
  LWA = 58,
  STD = 62
};

constexpr uint32_t NOP = 0x60000000;

enum class PPCLegacyInsn : uint32_t {
  NOINSN = 0,
  // Loads.
  LBZ = 0x88000000,
  LHZ = 0xa0000000,
  LWZ = 0x80000000,
  LHA = 0xa8000000,
  LWA = 0xe8000002,
  LD = 0xe8000000,
  LFS = 0xC0000000,
  LXSSP = 0xe4000003,
  LFD = 0xc8000000,
  LXSD = 0xe4000002,
  LXV = 0xf4000001,
  LXVP = 0x18000000,

  // Stores.
  STB = 0x98000000,
  STH = 0xb0000000,
  STW = 0x90000000,
  STD = 0xf8000000,
  STFS = 0xd0000000,
  STXSSP = 0xf4000003,
  STFD = 0xd8000000,
  STXSD = 0xf4000002,
  STXV = 0xf4000005,
  STXVP = 0x18000001
};
enum class PPCPrefixedInsn : uint64_t {
  NOINSN = 0,
  PREFIX_MLS = 0x0610000000000000,
  PREFIX_8LS = 0x0410000000000000,

  // Loads.
  PLBZ = PREFIX_MLS,
  PLHZ = PREFIX_MLS,
  PLWZ = PREFIX_MLS,
  PLHA = PREFIX_MLS,
  PLWA = PREFIX_8LS | 0xa4000000,
  PLD = PREFIX_8LS | 0xe4000000,
  PLFS = PREFIX_MLS,
  PLXSSP = PREFIX_8LS | 0xac000000,
  PLFD = PREFIX_MLS,
  PLXSD = PREFIX_8LS | 0xa8000000,
  PLXV = PREFIX_8LS | 0xc8000000,
  PLXVP = PREFIX_8LS | 0xe8000000,

  // Stores.
  PSTB = PREFIX_MLS,
  PSTH = PREFIX_MLS,
  PSTW = PREFIX_MLS,
  PSTD = PREFIX_8LS | 0xf4000000,
  PSTFS = PREFIX_MLS,
  PSTXSSP = PREFIX_8LS | 0xbc000000,
  PSTFD = PREFIX_MLS,
  PSTXSD = PREFIX_8LS | 0xb8000000,
  PSTXV = PREFIX_8LS | 0xd8000000,
  PSTXVP = PREFIX_8LS | 0xf8000000
};

static bool checkPPCLegacyInsn(uint32_t encoding) {
  PPCLegacyInsn insn = static_cast<PPCLegacyInsn>(encoding);
  if (insn == PPCLegacyInsn::NOINSN)
    return false;
#define PCREL_OPT(Legacy, PCRel, InsnMask)                                     \
  if (insn == PPCLegacyInsn::Legacy)                                           \
    return true;
#include "PPCInsns.def"
#undef PCREL_OPT
  return false;
}

// Masks to apply to legacy instructions when converting them to prefixed,
// pc-relative versions. For the most part, the primary opcode is shared
// between the legacy instruction and the suffix of its prefixed version.
// However, there are some instances where that isn't the case (DS-Form and
// DQ-form instructions).
enum class LegacyToPrefixMask : uint64_t {
  NOMASK = 0x0,
  OPC_AND_RST = 0xffe00000, // Primary opc (0-5) and R[ST] (6-10).
  ONLY_RST = 0x3e00000,     // [RS]T (6-10).
  ST_STX28_TO5 =
      0x8000000003e00000, // S/T (6-10) - The [S/T]X bit moves from 28 to 5.
};

class PPC64 final : public TargetInfo {
public:
  PPC64(Ctx &);
  int getTlsGdRelaxSkip(RelType type) const override;
  uint32_t calcEFlags() const override;
  RelExpr getRelExpr(RelType type, const Symbol &s,
                     const uint8_t *loc) const override;
  RelType getDynRel(RelType type) const override;
  int64_t getImplicitAddend(const uint8_t *buf, RelType type) const override;
  void writePltHeader(uint8_t *buf) const override;
  void writePlt(uint8_t *buf, const Symbol &sym,
                uint64_t pltEntryAddr) const override;
  void writeIplt(uint8_t *buf, const Symbol &sym,
                 uint64_t pltEntryAddr) const override;
  template <class ELFT, class RelTy>
  void scanSectionImpl(InputSectionBase &, Relocs<RelTy>);
  template <class ELFT> void scanSection1(InputSectionBase &);
  void scanSection(InputSectionBase &) override;
  void relocate(uint8_t *loc, const Relocation &rel,
                uint64_t val) const override;
  void writeGotHeader(uint8_t *buf) const override;
  bool needsThunk(RelExpr expr, RelType type, const InputFile *file,
                  uint64_t branchAddr, const Symbol &s,
                  int64_t a) const override;
  uint32_t getThunkSectionSpacing() const override;
  bool inBranchRange(RelType type, uint64_t src, uint64_t dst) const override;
  RelExpr adjustTlsExpr(RelType type, RelExpr expr) const override;
  RelExpr adjustGotPcExpr(RelType type, int64_t addend,
                          const uint8_t *loc) const override;
  void relaxGot(uint8_t *loc, const Relocation &rel, uint64_t val) const;
  void relocateAlloc(InputSection &sec, uint8_t *buf) const override;

  bool adjustPrologueForCrossSplitStack(uint8_t *loc, uint8_t *end,
                                        uint8_t stOther) const override;

private:
  void relaxTlsGdToIe(uint8_t *loc, const Relocation &rel, uint64_t val) const;
  void relaxTlsGdToLe(uint8_t *loc, const Relocation &rel, uint64_t val) const;
  void relaxTlsLdToLe(uint8_t *loc, const Relocation &rel, uint64_t val) const;
  void relaxTlsIeToLe(uint8_t *loc, const Relocation &rel, uint64_t val) const;
};
} // namespace

uint64_t elf::getPPC64TocBase(Ctx &ctx) {
  // The TOC consists of sections .got, .toc, .tocbss, .plt in that order. The
  // TOC starts where the first of these sections starts. We always create a
  // .got when we see a relocation that uses it, so for us the start is always
  // the .got.
  uint64_t tocVA = ctx.in.got->getVA();

  // Per the ppc64-elf-linux ABI, The TOC base is TOC value plus 0x8000
  // thus permitting a full 64 Kbytes segment. Note that the glibc startup
  // code (crt1.o) assumes that you can get from the TOC base to the
  // start of the .toc section with only a single (signed) 16-bit relocation.
  return tocVA + ppc64TocOffset;
}

unsigned elf::getPPC64GlobalEntryToLocalEntryOffset(Ctx &ctx, uint8_t stOther) {
  // The offset is encoded into the 3 most significant bits of the st_other
  // field, with some special values described in section 3.4.1 of the ABI:
  // 0   --> Zero offset between the GEP and LEP, and the function does NOT use
  //         the TOC pointer (r2). r2 will hold the same value on returning from
  //         the function as it did on entering the function.
  // 1   --> Zero offset between the GEP and LEP, and r2 should be treated as a
  //         caller-saved register for all callers.
  // 2-6 --> The  binary logarithm of the offset eg:
  //         2 --> 2^2 = 4 bytes -->  1 instruction.
  //         6 --> 2^6 = 64 bytes --> 16 instructions.
  // 7   --> Reserved.
  uint8_t gepToLep = (stOther >> 5) & 7;
  if (gepToLep < 2)
    return 0;

  // The value encoded in the st_other bits is the
  // log-base-2(offset).
  if (gepToLep < 7)
    return 1 << gepToLep;

  ErrAlways(ctx)
      << "reserved value of 7 in the 3 most-significant-bits of st_other";
  return 0;
}

void elf::writePrefixedInst(Ctx &ctx, uint8_t *loc, uint64_t insn) {
  insn = ctx.arg.isLE ? insn << 32 | insn >> 32 : insn;
  write64(ctx, loc, insn);
}

static bool addOptional(Ctx &ctx, StringRef name, uint64_t value,
                        std::vector<Defined *> &defined) {
  Symbol *sym = ctx.symtab->find(name);
  if (!sym || sym->isDefined())
    return false;
  sym->resolve(ctx, Defined{ctx, ctx.internalFile, StringRef(), STB_GLOBAL,
                            STV_HIDDEN, STT_FUNC, value,
                            /*size=*/0, /*section=*/nullptr});
  defined.push_back(cast<Defined>(sym));
  return true;
}

// If from is 14, write ${prefix}14: firstInsn; ${prefix}15:
// firstInsn+0x200008; ...; ${prefix}31: firstInsn+(31-14)*0x200008; $tail
// The labels are defined only if they exist in the symbol table.
static void writeSequence(Ctx &ctx, const char *prefix, int from,
                          uint32_t firstInsn, ArrayRef<uint32_t> tail) {
  std::vector<Defined *> defined;
  char name[16];
  int first;
  const size_t size = 32 - from + tail.size();
  MutableArrayRef<uint32_t> buf(ctx.bAlloc.Allocate<uint32_t>(size), size);
  uint32_t *ptr = buf.data();
  for (int r = from; r < 32; ++r) {
    format("%s%d", prefix, r).snprint(name, sizeof(name));
    if (addOptional(ctx, name, 4 * (r - from), defined) && defined.size() == 1)
      first = r - from;
    write32(ctx, ptr++, firstInsn + 0x200008 * (r - from));
  }
  for (uint32_t insn : tail)
    write32(ctx, ptr++, insn);
  assert(ptr == &*buf.end());

  if (defined.empty())
    return;
  // The full section content has the extent of [begin, end). We drop unused
  // instructions and write [first,end).
  auto *sec = make<InputSection>(
      ctx.internalFile, ".text", SHT_PROGBITS, SHF_ALLOC, /*addralign=*/4,
      /*entsize=*/0,
      ArrayRef(reinterpret_cast<uint8_t *>(buf.data() + first),
               4 * (buf.size() - first)));
  ctx.inputSections.push_back(sec);
  for (Defined *sym : defined) {
    sym->section = sec;
    sym->value -= 4 * first;
  }
}

// Implements some save and restore functions as described by ELF V2 ABI to be
// compatible with GCC. With GCC -Os, when the number of call-saved registers
// exceeds a certain threshold, GCC generates _savegpr0_* _restgpr0_* calls and
// expects the linker to define them. See
// https://sourceware.org/pipermail/binutils/2002-February/017444.html and
// https://sourceware.org/pipermail/binutils/2004-August/036765.html . This is
// weird because libgcc.a would be the natural place. The linker generation
// approach has the advantage that the linker can generate multiple copies to
// avoid long branch thunks. However, we don't consider the advantage
// significant enough to complicate our trunk implementation, so we take the
// simple approach and synthesize .text sections providing the implementation.
void elf::addPPC64SaveRestore(Ctx &ctx) {
  constexpr uint32_t blr = 0x4e800020, mtlr_0 = 0x7c0803a6;

  // _restgpr0_14: ld 14, -144(1); _restgpr0_15: ld 15, -136(1); ...
  // Tail: ld 0, 16(1); mtlr 0; blr
  writeSequence(ctx, "_restgpr0_", 14, 0xe9c1ff70, {0xe8010010, mtlr_0, blr});
  // _restgpr1_14: ld 14, -144(12); _restgpr1_15: ld 15, -136(12); ...
  // Tail: blr
  writeSequence(ctx, "_restgpr1_", 14, 0xe9ccff70, {blr});
  // _savegpr0_14: std 14, -144(1); _savegpr0_15: std 15, -136(1); ...
  // Tail: std 0, 16(1); blr
  writeSequence(ctx, "_savegpr0_", 14, 0xf9c1ff70, {0xf8010010, blr});
  // _savegpr1_14: std 14, -144(12); _savegpr1_15: std 15, -136(12); ...
  // Tail: blr
  writeSequence(ctx, "_savegpr1_", 14, 0xf9ccff70, {blr});
}

// Find the R_PPC64_ADDR64 in .rela.toc with matching offset.
template <typename ELFT>
static std::pair<Defined *, int64_t>
getRelaTocSymAndAddend(InputSectionBase *tocSec, uint64_t offset) {
  // .rela.toc contains exclusively R_PPC64_ADDR64 relocations sorted by
  // r_offset: 0, 8, 16, etc. For a given Offset, Offset / 8 gives us the
  // relocation index in most cases.
  //
  // In rare cases a TOC entry may store a constant that doesn't need an
  // R_PPC64_ADDR64, the corresponding r_offset is therefore missing. Offset / 8
  // points to a relocation with larger r_offset. Do a linear probe then.
  // Constants are extremely uncommon in .toc and the extra number of array
  // accesses can be seen as a small constant.
  ArrayRef<typename ELFT::Rela> relas =
      tocSec->template relsOrRelas<ELFT>().relas;
  if (relas.empty())
    return {};
  uint64_t index = std::min<uint64_t>(offset / 8, relas.size() - 1);
  for (;;) {
    if (relas[index].r_offset == offset) {
      Symbol &sym = tocSec->file->getRelocTargetSym(relas[index]);
      return {dyn_cast<Defined>(&sym), getAddend<ELFT>(relas[index])};
    }
    if (relas[index].r_offset < offset || index == 0)
      break;
    --index;
  }
  return {};
}

// When accessing a symbol defined in another translation unit, compilers
// reserve a .toc entry, allocate a local label and generate toc-indirect
// instructions:
//
//   addis 3, 2, .LC0@toc@ha  # R_PPC64_TOC16_HA
//   ld    3, .LC0@toc@l(3)   # R_PPC64_TOC16_LO_DS, load the address from a .toc entry
//   ld/lwa 3, 0(3)           # load the value from the address
//
//   .section .toc,"aw",@progbits
//   .LC0: .tc var[TC],var
//
// If var is defined, non-preemptable and addressable with a 32-bit signed
// offset from the toc base, the address of var can be computed by adding an
// offset to the toc base, saving a load.
//
//   addis 3,2,var@toc@ha     # this may be relaxed to a nop,
//   addi  3,3,var@toc@l      # then this becomes addi 3,2,var@toc
//   ld/lwa 3, 0(3)           # load the value from the address
//
// Returns true if the relaxation is performed.
static bool tryRelaxPPC64TocIndirection(Ctx &ctx, const Relocation &rel,
                                        uint8_t *bufLoc) {
  assert(ctx.arg.tocOptimize);
  if (rel.addend < 0)
    return false;

  // If the symbol is not the .toc section, this isn't a toc-indirection.
  Defined *defSym = dyn_cast<Defined>(rel.sym);
  if (!defSym || !defSym->isSection() || defSym->section->name != ".toc")
    return false;

  Defined *d;
  int64_t addend;
  auto *tocISB = cast<InputSectionBase>(defSym->section);
  std::tie(d, addend) =
      ctx.arg.isLE ? getRelaTocSymAndAddend<ELF64LE>(tocISB, rel.addend)
                   : getRelaTocSymAndAddend<ELF64BE>(tocISB, rel.addend);

  // Only non-preemptable defined symbols can be relaxed.
  if (!d || d->isPreemptible)
    return false;

  // R_PPC64_ADDR64 should have created a canonical PLT for the non-preemptable
  // ifunc and changed its type to STT_FUNC.
  assert(!d->isGnuIFunc());

  // Two instructions can materialize a 32-bit signed offset from the toc base.
  uint64_t tocRelative = d->getVA(ctx, addend) - getPPC64TocBase(ctx);
  if (!isInt<32>(tocRelative))
    return false;

  // Add PPC64TocOffset that will be subtracted by PPC64::relocate().
  static_cast<const PPC64 &>(*ctx.target)
      .relaxGot(bufLoc, rel, tocRelative + ppc64TocOffset);
  return true;
}

// Relocation masks following the #lo(value), #hi(value), #ha(value),
// #higher(value), #highera(value), #highest(value), and #highesta(value)
// macros defined in section 4.5.1. Relocation Types of the PPC-elf64abi
// document.
static uint16_t lo(uint64_t v) { return v; }
static uint16_t hi(uint64_t v) { return v >> 16; }
static uint64_t ha(uint64_t v) { return (v + 0x8000) >> 16; }
static uint16_t higher(uint64_t v) { return v >> 32; }
static uint16_t highera(uint64_t v) { return (v + 0x8000) >> 32; }
static uint16_t highest(uint64_t v) { return v >> 48; }
static uint16_t highesta(uint64_t v) { return (v + 0x8000) >> 48; }

// Extracts the 'PO' field of an instruction encoding.
static uint8_t getPrimaryOpCode(uint32_t encoding) { return (encoding >> 26); }

static bool isDQFormInstruction(uint32_t encoding) {
  switch (getPrimaryOpCode(encoding)) {
  default:
    return false;
  case 6: // Power10 paired loads/stores (lxvp, stxvp).
  case 56:
    // The only instruction with a primary opcode of 56 is `lq`.
    return true;
  case 61:
    // There are both DS and DQ instruction forms with this primary opcode.
    // Namely `lxv` and `stxv` are the DQ-forms that use it.
    // The DS 'XO' bits being set to 01 is restricted to DQ form.
    return (encoding & 3) == 0x1;
  }
}

static bool isDSFormInstruction(PPCLegacyInsn insn) {
  switch (insn) {
  default:
    return false;
  case PPCLegacyInsn::LWA:
  case PPCLegacyInsn::LD:
  case PPCLegacyInsn::LXSD:
  case PPCLegacyInsn::LXSSP:
  case PPCLegacyInsn::STD:
  case PPCLegacyInsn::STXSD:
  case PPCLegacyInsn::STXSSP:
    return true;
  }
}

static PPCLegacyInsn getPPCLegacyInsn(uint32_t encoding) {
  uint32_t opc = encoding & 0xfc000000;

  // If the primary opcode is shared between multiple instructions, we need to
  // fix it up to match the actual instruction we are after.
  if ((opc == 0xe4000000 || opc == 0xe8000000 || opc == 0xf4000000 ||
       opc == 0xf8000000) &&
      !isDQFormInstruction(encoding))
    opc = encoding & 0xfc000003;
  else if (opc == 0xf4000000)
    opc = encoding & 0xfc000007;
  else if (opc == 0x18000000)
    opc = encoding & 0xfc00000f;

  // If the value is not one of the enumerators in PPCLegacyInsn, we want to
  // return PPCLegacyInsn::NOINSN.
  if (!checkPPCLegacyInsn(opc))
    return PPCLegacyInsn::NOINSN;
  return static_cast<PPCLegacyInsn>(opc);
}

static PPCPrefixedInsn getPCRelativeForm(PPCLegacyInsn insn) {
  switch (insn) {
#define PCREL_OPT(Legacy, PCRel, InsnMask)                                     \
  case PPCLegacyInsn::Legacy:                                                  \
    return PPCPrefixedInsn::PCRel
#include "PPCInsns.def"
#undef PCREL_OPT
  }
  return PPCPrefixedInsn::NOINSN;
}

static LegacyToPrefixMask getInsnMask(PPCLegacyInsn insn) {
  switch (insn) {
#define PCREL_OPT(Legacy, PCRel, InsnMask)                                     \
  case PPCLegacyInsn::Legacy:                                                  \
    return LegacyToPrefixMask::InsnMask
#include "PPCInsns.def"
#undef PCREL_OPT
  }
  return LegacyToPrefixMask::NOMASK;
}
static uint64_t getPCRelativeForm(uint32_t encoding) {
  PPCLegacyInsn origInsn = getPPCLegacyInsn(encoding);
  PPCPrefixedInsn pcrelInsn = getPCRelativeForm(origInsn);
  if (pcrelInsn == PPCPrefixedInsn::NOINSN)
    return UINT64_C(-1);
  LegacyToPrefixMask origInsnMask = getInsnMask(origInsn);
  uint64_t pcrelEncoding =
      (uint64_t)pcrelInsn | (encoding & (uint64_t)origInsnMask);

  // If the mask requires moving bit 28 to bit 5, do that now.
  if (origInsnMask == LegacyToPrefixMask::ST_STX28_TO5)
    pcrelEncoding |= (encoding & 0x8) << 23;
  return pcrelEncoding;
}

static bool isInstructionUpdateForm(uint32_t encoding) {
  switch (getPrimaryOpCode(encoding)) {
  default:
    return false;
  case LBZU:
  case LHAU:
  case LHZU:
  case LWZU:
  case LFSU:
  case LFDU:
  case STBU:
  case STHU:
  case STWU:
  case STFSU:
  case STFDU:
    return true;
    // LWA has the same opcode as LD, and the DS bits is what differentiates
    // between LD/LDU/LWA
  case LD:
  case STD:
    return (encoding & 3) == 1;
  }
}

// Compute the total displacement between the prefixed instruction that gets
// to the start of the data and the load/store instruction that has the offset
// into the data structure.
// For example:
// paddi 3, 0, 1000, 1
// lwz 3, 20(3)
// Should add up to 1020 for total displacement.
static int64_t getTotalDisp(uint64_t prefixedInsn, uint32_t accessInsn) {
  int64_t disp34 = llvm::SignExtend64(
      ((prefixedInsn & 0x3ffff00000000) >> 16) | (prefixedInsn & 0xffff), 34);
  int32_t disp16 = llvm::SignExtend32(accessInsn & 0xffff, 16);
  // For DS and DQ form instructions, we need to mask out the XO bits.
  if (isDQFormInstruction(accessInsn))
    disp16 &= ~0xf;
  else if (isDSFormInstruction(getPPCLegacyInsn(accessInsn)))
    disp16 &= ~0x3;
  return disp34 + disp16;
}

// There are a number of places when we either want to read or write an
// instruction when handling a half16 relocation type. On big-endian the buffer
// pointer is pointing into the middle of the word we want to extract, and on
// little-endian it is pointing to the start of the word. These 2 helpers are to
// simplify reading and writing in that context.
static void writeFromHalf16(Ctx &ctx, uint8_t *loc, uint32_t insn) {
  write32(ctx, ctx.arg.isLE ? loc : loc - 2, insn);
}

static uint32_t readFromHalf16(Ctx &ctx, const uint8_t *loc) {
  return read32(ctx, ctx.arg.isLE ? loc : loc - 2);
}

static uint64_t readPrefixedInst(Ctx &ctx, const uint8_t *loc) {
  uint64_t fullInstr = read64(ctx, loc);
  return ctx.arg.isLE ? (fullInstr << 32 | fullInstr >> 32) : fullInstr;
}

PPC64::PPC64(Ctx &ctx) : TargetInfo(ctx) {
  copyRel = R_PPC64_COPY;
  gotRel = R_PPC64_GLOB_DAT;
  pltRel = R_PPC64_JMP_SLOT;
  relativeRel = R_PPC64_RELATIVE;
  iRelativeRel = R_PPC64_IRELATIVE;
  symbolicRel = R_PPC64_ADDR64;
  pltHeaderSize = 60;
  pltEntrySize = 4;
  ipltEntrySize = 16; // PPC64PltCallStub::size
  gotHeaderEntriesNum = 1;
  gotPltHeaderEntriesNum = 2;
  needsThunks = true;

  tlsModuleIndexRel = R_PPC64_DTPMOD64;
  tlsOffsetRel = R_PPC64_DTPREL64;

  tlsGotRel = R_PPC64_TPREL64;

  needsMoreStackNonSplit = false;

  // We need 64K pages (at least under glibc/Linux, the loader won't
  // set different permissions on a finer granularity than that).
  defaultMaxPageSize = 65536;

  // The PPC64 ELF ABI v1 spec, says:
  //
  //   It is normally desirable to put segments with different characteristics
  //   in separate 256 Mbyte portions of the address space, to give the
  //   operating system full paging flexibility in the 64-bit address space.
  //
  // And because the lowest non-zero 256M boundary is 0x10000000, PPC64 linkers
  // use 0x10000000 as the starting address.
  defaultImageBase = 0x10000000;

  write32(ctx, trapInstr.data(), 0x7fe00008);
}

int PPC64::getTlsGdRelaxSkip(RelType type) const {
  // A __tls_get_addr call instruction is marked with 2 relocations:
  //
  //   R_PPC64_TLSGD / R_PPC64_TLSLD: marker relocation
  //   R_PPC64_REL24: __tls_get_addr
  //
  // After the relaxation we no longer call __tls_get_addr and should skip both
  // relocations to not create a false dependence on __tls_get_addr being
  // defined.
  if (type == R_PPC64_TLSGD || type == R_PPC64_TLSLD)
    return 2;
  return 1;
}

static uint32_t getEFlags(InputFile *file) {
  if (file->ekind == ELF64BEKind)
    return cast<ObjFile<ELF64BE>>(file)->getObj().getHeader().e_flags;
  return cast<ObjFile<ELF64LE>>(file)->getObj().getHeader().e_flags;
}

// This file implements v2 ABI. This function makes sure that all
// object files have v2 or an unspecified version as an ABI version.
uint32_t PPC64::calcEFlags() const {
  for (InputFile *f : ctx.objectFiles) {
    uint32_t flag = getEFlags(f);
    if (flag == 1)
      ErrAlways(ctx) << f << ": ABI version 1 is not supported";
    else if (flag > 2)
      ErrAlways(ctx) << f << ": unrecognized e_flags: " << flag;
  }
  return 2;
}

void PPC64::relaxGot(uint8_t *loc, const Relocation &rel, uint64_t val) const {
  switch (rel.type) {
  case R_PPC64_TOC16_HA:
    // Convert "addis reg, 2, .LC0@toc@h" to "addis reg, 2, var@toc@h" or "nop".
    relocate(loc, rel, val);
    break;
  case R_PPC64_TOC16_LO_DS: {
    // Convert "ld reg, .LC0@toc@l(reg)" to "addi reg, reg, var@toc@l" or
    // "addi reg, 2, var@toc".
    uint32_t insn = readFromHalf16(ctx, loc);
    if (getPrimaryOpCode(insn) != LD)
      ErrAlways(ctx)
          << "expected a 'ld' for got-indirect to toc-relative relaxing";
    writeFromHalf16(ctx, loc, (insn & 0x03ffffff) | 0x38000000);
    relocateNoSym(loc, R_PPC64_TOC16_LO, val);
    break;
  }
  case R_PPC64_GOT_PCREL34: {
    // Clear the first 8 bits of the prefix and the first 6 bits of the
    // instruction (the primary opcode).
    uint64_t insn = readPrefixedInst(ctx, loc);
    if ((insn & 0xfc000000) != 0xe4000000)
      ErrAlways(ctx)
          << "expected a 'pld' for got-indirect to pc-relative relaxing";
    insn &= ~0xff000000fc000000;

    // Replace the cleared bits with the values for PADDI (0x600000038000000);
    insn |= 0x600000038000000;
    writePrefixedInst(ctx, loc, insn);
    relocate(loc, rel, val);
    break;
  }
  case R_PPC64_PCREL_OPT: {
    // We can only relax this if the R_PPC64_GOT_PCREL34 at this offset can
    // be relaxed. The eligibility for the relaxation needs to be determined
    // on that relocation since this one does not relocate a symbol.
    uint64_t insn = readPrefixedInst(ctx, loc);
    uint32_t accessInsn = read32(ctx, loc + rel.addend);
    uint64_t pcRelInsn = getPCRelativeForm(accessInsn);

    // This error is not necessary for correctness but is emitted for now
    // to ensure we don't miss these opportunities in real code. It can be
    // removed at a later date.
    if (pcRelInsn == UINT64_C(-1)) {
      Err(ctx)
          << "unrecognized instruction for R_PPC64_PCREL_OPT relaxation: 0x"
          << utohexstr(accessInsn, true);
      break;
    }

    int64_t totalDisp = getTotalDisp(insn, accessInsn);
    if (!isInt<34>(totalDisp))
      break; // Displacement doesn't fit.
    // Convert the PADDI to the prefixed version of accessInsn and convert
    // accessInsn to a nop.
    writePrefixedInst(ctx, loc,
                      pcRelInsn | ((totalDisp & 0x3ffff0000) << 16) |
                          (totalDisp & 0xffff));
    write32(ctx, loc + rel.addend, NOP); // nop accessInsn.
    break;
  }
  default:
    llvm_unreachable("unexpected relocation type");
  }
}

void PPC64::relaxTlsGdToLe(uint8_t *loc, const Relocation &rel,
                           uint64_t val) const {
  // Reference: 3.7.4.2 of the 64-bit ELF V2 abi supplement.
  // The general dynamic code sequence for a global `x` will look like:
  // Instruction                    Relocation                Symbol
  // addis r3, r2, x@got@tlsgd@ha   R_PPC64_GOT_TLSGD16_HA      x
  // addi  r3, r3, x@got@tlsgd@l    R_PPC64_GOT_TLSGD16_LO      x
  // bl __tls_get_addr(x@tlsgd)     R_PPC64_TLSGD               x
  //                                R_PPC64_REL24               __tls_get_addr
  // nop                            None                       None

  // Relaxing to local exec entails converting:
  // addis r3, r2, x@got@tlsgd@ha    into      nop
  // addi  r3, r3, x@got@tlsgd@l     into      addis r3, r13, x@tprel@ha
  // bl __tls_get_addr(x@tlsgd)      into      nop
  // nop                             into      addi r3, r3, x@tprel@l

  switch (rel.type) {
  case R_PPC64_GOT_TLSGD16_HA:
    writeFromHalf16(ctx, loc, NOP);
    break;
  case R_PPC64_GOT_TLSGD16:
  case R_PPC64_GOT_TLSGD16_LO:
    writeFromHalf16(ctx, loc, 0x3c6d0000); // addis r3, r13
    relocateNoSym(loc, R_PPC64_TPREL16_HA, val);
    break;
  case R_PPC64_GOT_TLSGD_PCREL34:
    // Relax from paddi r3, 0, x@got@tlsgd@pcrel, 1 to
    //            paddi r3, r13, x@tprel, 0
    writePrefixedInst(ctx, loc, 0x06000000386d0000);
    relocateNoSym(loc, R_PPC64_TPREL34, val);
    break;
  case R_PPC64_TLSGD: {
    // PC Relative Relaxation:
    // Relax from bl __tls_get_addr@notoc(x@tlsgd) to
    //            nop
    // TOC Relaxation:
    // Relax from bl __tls_get_addr(x@tlsgd)
    //            nop
    // to
    //            nop
    //            addi r3, r3, x@tprel@l
    const uintptr_t locAsInt = reinterpret_cast<uintptr_t>(loc);
    if (locAsInt % 4 == 0) {
      write32(ctx, loc, NOP);            // nop
      write32(ctx, loc + 4, 0x38630000); // addi r3, r3
      // Since we are relocating a half16 type relocation and Loc + 4 points to
      // the start of an instruction we need to advance the buffer by an extra
      // 2 bytes on BE.
      relocateNoSym(loc + 4 + (ctx.arg.ekind == ELF64BEKind ? 2 : 0),
                    R_PPC64_TPREL16_LO, val);
    } else if (locAsInt % 4 == 1) {
      write32(ctx, loc - 1, NOP);
    } else {
      Err(ctx) << "R_PPC64_TLSGD has unexpected byte alignment";
    }
    break;
  }
  default:
    llvm_unreachable("unsupported relocation for TLS GD to LE relaxation");
  }
}

void PPC64::relaxTlsLdToLe(uint8_t *loc, const Relocation &rel,
                           uint64_t val) const {
  // Reference: 3.7.4.3 of the 64-bit ELF V2 abi supplement.
  // The local dynamic code sequence for a global `x` will look like:
  // Instruction                    Relocation                Symbol
  // addis r3, r2, x@got@tlsld@ha   R_PPC64_GOT_TLSLD16_HA      x
  // addi  r3, r3, x@got@tlsld@l    R_PPC64_GOT_TLSLD16_LO      x
  // bl __tls_get_addr(x@tlsgd)     R_PPC64_TLSLD               x
  //                                R_PPC64_REL24               __tls_get_addr
  // nop                            None                       None

  // Relaxing to local exec entails converting:
  // addis r3, r2, x@got@tlsld@ha   into      nop
  // addi  r3, r3, x@got@tlsld@l    into      addis r3, r13, 0
  // bl __tls_get_addr(x@tlsgd)     into      nop
  // nop                            into      addi r3, r3, 4096

  switch (rel.type) {
  case R_PPC64_GOT_TLSLD16_HA:
    writeFromHalf16(ctx, loc, NOP);
    break;
  case R_PPC64_GOT_TLSLD16_LO:
    writeFromHalf16(ctx, loc, 0x3c6d0000); // addis r3, r13, 0
    break;
  case R_PPC64_GOT_TLSLD_PCREL34:
    // Relax from paddi r3, 0, x1@got@tlsld@pcrel, 1 to
    //            paddi r3, r13, 0x1000, 0
    writePrefixedInst(ctx, loc, 0x06000000386d1000);
    break;
  case R_PPC64_TLSLD: {
    // PC Relative Relaxation:
    // Relax from bl __tls_get_addr@notoc(x@tlsld)
    // to
    //            nop
    // TOC Relaxation:
    // Relax from bl __tls_get_addr(x@tlsld)
    //            nop
    // to
    //            nop
    //            addi r3, r3, 4096
    const uintptr_t locAsInt = reinterpret_cast<uintptr_t>(loc);
    if (locAsInt % 4 == 0) {
      write32(ctx, loc, NOP);
      write32(ctx, loc + 4, 0x38631000); // addi r3, r3, 4096
    } else if (locAsInt % 4 == 1) {
      write32(ctx, loc - 1, NOP);
    } else {
      Err(ctx) << "R_PPC64_TLSLD has unexpected byte alignment";
    }
    break;
  }
  case R_PPC64_DTPREL16:
  case R_PPC64_DTPREL16_HA:
  case R_PPC64_DTPREL16_HI:
  case R_PPC64_DTPREL16_DS:
  case R_PPC64_DTPREL16_LO:
  case R_PPC64_DTPREL16_LO_DS:
  case R_PPC64_DTPREL34:
    relocate(loc, rel, val);
    break;
  default:
    llvm_unreachable("unsupported relocation for TLS LD to LE relaxation");
  }
}

// Map X-Form instructions to their DS-Form counterparts, if applicable.
// The full encoding is returned here to distinguish between the different
// DS-Form instructions.
unsigned elf::getPPCDSFormOp(unsigned secondaryOp) {
  switch (secondaryOp) {
  case LWAX:
    return (LWA << 26) | 0x2;
  case LDX:
    return LD << 26;
  case STDX:
    return STD << 26;
  default:
    return 0;
  }
}

unsigned elf::getPPCDFormOp(unsigned secondaryOp) {
  switch (secondaryOp) {
  case LBZX:
    return LBZ << 26;
  case LHZX:
    return LHZ << 26;
  case LWZX:
    return LWZ << 26;
  case STBX:
    return STB << 26;
  case STHX:
    return STH << 26;
  case STWX:
    return STW << 26;
  case LHAX:
    return LHA << 26;
  case LFSX:
    return LFS << 26;
  case LFDX:
    return LFD << 26;
  case STFSX:
    return STFS << 26;
  case STFDX:
    return STFD << 26;
  case ADD:
    return ADDI << 26;
  default:
    return 0;
  }
}

void PPC64::relaxTlsIeToLe(uint8_t *loc, const Relocation &rel,
                           uint64_t val) const {
  // The initial exec code sequence for a global `x` will look like:
  // Instruction                    Relocation                Symbol
  // addis r9, r2, x@got@tprel@ha   R_PPC64_GOT_TPREL16_HA      x
  // ld    r9, x@got@tprel@l(r9)    R_PPC64_GOT_TPREL16_LO_DS   x
  // add r9, r9, x@tls              R_PPC64_TLS                 x

  // Relaxing to local exec entails converting:
  // addis r9, r2, x@got@tprel@ha       into        nop
  // ld r9, x@got@tprel@l(r9)           into        addis r9, r13, x@tprel@ha
  // add r9, r9, x@tls                  into        addi r9, r9, x@tprel@l

  // x@tls R_PPC64_TLS is a relocation which does not compute anything,
  // it is replaced with r13 (thread pointer).

  // The add instruction in the initial exec sequence has multiple variations
  // that need to be handled. If we are building an address it will use an add
  // instruction, if we are accessing memory it will use any of the X-form
  // indexed load or store instructions.

  unsigned offset = (ctx.arg.ekind == ELF64BEKind) ? 2 : 0;
  switch (rel.type) {
  case R_PPC64_GOT_TPREL16_HA:
    write32(ctx, loc - offset, NOP);
    break;
  case R_PPC64_GOT_TPREL16_LO_DS:
  case R_PPC64_GOT_TPREL16_DS: {
    uint32_t regNo = read32(ctx, loc - offset) & 0x03e00000; // bits 6-10
    write32(ctx, loc - offset, 0x3c0d0000 | regNo);          // addis RegNo, r13
    relocateNoSym(loc, R_PPC64_TPREL16_HA, val);
    break;
  }
  case R_PPC64_GOT_TPREL_PCREL34: {
    const uint64_t pldRT = readPrefixedInst(ctx, loc) & 0x0000000003e00000;
    // paddi RT(from pld), r13, symbol@tprel, 0
    writePrefixedInst(ctx, loc, 0x06000000380d0000 | pldRT);
    relocateNoSym(loc, R_PPC64_TPREL34, val);
    break;
  }
  case R_PPC64_TLS: {
    const uintptr_t locAsInt = reinterpret_cast<uintptr_t>(loc);
    if (locAsInt % 4 == 0) {
      uint32_t primaryOp = getPrimaryOpCode(read32(ctx, loc));
      if (primaryOp != 31)
        ErrAlways(ctx) << "unrecognized instruction for IE to LE R_PPC64_TLS";
      uint32_t secondaryOp = (read32(ctx, loc) & 0x000007fe) >> 1; // bits 21-30
      uint32_t dFormOp = getPPCDFormOp(secondaryOp);
      uint32_t finalReloc;
      if (dFormOp == 0) { // Expecting a DS-Form instruction.
        dFormOp = getPPCDSFormOp(secondaryOp);
        if (dFormOp == 0)
          ErrAlways(ctx) << "unrecognized instruction for IE to LE R_PPC64_TLS";
        finalReloc = R_PPC64_TPREL16_LO_DS;
      } else
        finalReloc = R_PPC64_TPREL16_LO;
      write32(ctx, loc, dFormOp | (read32(ctx, loc) & 0x03ff0000));
      relocateNoSym(loc + offset, finalReloc, val);
    } else if (locAsInt % 4 == 1) {
      // If the offset is not 4 byte aligned then we have a PCRel type reloc.
      // This version of the relocation is offset by one byte from the
      // instruction it references.
      uint32_t tlsInstr = read32(ctx, loc - 1);
      uint32_t primaryOp = getPrimaryOpCode(tlsInstr);
      if (primaryOp != 31)
        Err(ctx) << "unrecognized instruction for IE to LE R_PPC64_TLS";
      uint32_t secondaryOp = (tlsInstr & 0x000007FE) >> 1; // bits 21-30
      // The add is a special case and should be turned into a nop. The paddi
      // that comes before it will already have computed the address of the
      // symbol.
      if (secondaryOp == 266) {
        // Check if the add uses the same result register as the input register.
        uint32_t rt = (tlsInstr & 0x03E00000) >> 21; // bits 6-10
        uint32_t ra = (tlsInstr & 0x001F0000) >> 16; // bits 11-15
        if (ra == rt) {
          write32(ctx, loc - 1, NOP);
        } else {
          // mr rt, ra
          write32(ctx, loc - 1,
                  0x7C000378 | (rt << 16) | (ra << 21) | (ra << 11));
        }
      } else {
        uint32_t dFormOp = getPPCDFormOp(secondaryOp);
        if (dFormOp == 0) { // Expecting a DS-Form instruction.
          dFormOp = getPPCDSFormOp(secondaryOp);
          if (dFormOp == 0)
            Err(ctx) << "unrecognized instruction for IE to LE R_PPC64_TLS";
        }
        write32(ctx, loc - 1, (dFormOp | (tlsInstr & 0x03ff0000)));
      }
    } else {
      Err(ctx) << "R_PPC64_TLS must be either 4 byte aligned or one byte "
                  "offset from 4 byte aligned";
    }
    break;
  }
  default:
    llvm_unreachable("unknown relocation for IE to LE");
    break;
  }
}

RelExpr PPC64::getRelExpr(RelType type, const Symbol &s,
                          const uint8_t *loc) const {
  switch (type) {
  case R_PPC64_NONE:
    return R_NONE;
  case R_PPC64_ADDR16:
  case R_PPC64_ADDR16_DS:
  case R_PPC64_ADDR16_HA:
  case R_PPC64_ADDR16_HI:
  case R_PPC64_ADDR16_HIGH:
  case R_PPC64_ADDR16_HIGHER:
  case R_PPC64_ADDR16_HIGHERA:
  case R_PPC64_ADDR16_HIGHEST:
  case R_PPC64_ADDR16_HIGHESTA:
  case R_PPC64_ADDR16_LO:
  case R_PPC64_ADDR16_LO_DS:
  case R_PPC64_ADDR32:
  case R_PPC64_ADDR64:
    return R_ABS;
  case R_PPC64_GOT16:
  case R_PPC64_GOT16_DS:
  case R_PPC64_GOT16_HA:
  case R_PPC64_GOT16_HI:
  case R_PPC64_GOT16_LO:
  case R_PPC64_GOT16_LO_DS:
    return R_GOT_OFF;
  case R_PPC64_TOC16:
  case R_PPC64_TOC16_DS:
  case R_PPC64_TOC16_HI:
  case R_PPC64_TOC16_LO:
    return R_GOTREL;
  case R_PPC64_GOT_PCREL34:
  case R_PPC64_GOT_TPREL_PCREL34:
  case R_PPC64_PCREL_OPT:
    return R_GOT_PC;
  case R_PPC64_TOC16_HA:
  case R_PPC64_TOC16_LO_DS:
    return ctx.arg.tocOptimize ? RE_PPC64_RELAX_TOC : R_GOTREL;
  case R_PPC64_TOC:
    return RE_PPC64_TOCBASE;
  case R_PPC64_REL14:
  case R_PPC64_REL24:
    return RE_PPC64_CALL_PLT;
  case R_PPC64_REL24_NOTOC:
    return R_PLT_PC;
  case R_PPC64_REL16_LO:
  case R_PPC64_REL16_HA:
  case R_PPC64_REL16_HI:
  case R_PPC64_REL32:
  case R_PPC64_REL64:
  case R_PPC64_PCREL34:
    return R_PC;
  case R_PPC64_GOT_TLSGD16:
  case R_PPC64_GOT_TLSGD16_HA:
  case R_PPC64_GOT_TLSGD16_HI:
  case R_PPC64_GOT_TLSGD16_LO:
    return R_TLSGD_GOT;
  case R_PPC64_GOT_TLSGD_PCREL34:
    return R_TLSGD_PC;
  case R_PPC64_GOT_TLSLD16:
  case R_PPC64_GOT_TLSLD16_HA:
  case R_PPC64_GOT_TLSLD16_HI:
  case R_PPC64_GOT_TLSLD16_LO:
    return R_TLSLD_GOT;
  case R_PPC64_GOT_TLSLD_PCREL34:
    return R_TLSLD_PC;
  case R_PPC64_GOT_TPREL16_HA:
  case R_PPC64_GOT_TPREL16_LO_DS:
  case R_PPC64_GOT_TPREL16_DS:
  case R_PPC64_GOT_TPREL16_HI:
    return R_GOT_OFF;
  case R_PPC64_GOT_DTPREL16_HA:
  case R_PPC64_GOT_DTPREL16_LO_DS:
  case R_PPC64_GOT_DTPREL16_DS:
  case R_PPC64_GOT_DTPREL16_HI:
    return R_TLSLD_GOT_OFF;
  case R_PPC64_TPREL16:
  case R_PPC64_TPREL16_HA:
  case R_PPC64_TPREL16_LO:
  case R_PPC64_TPREL16_HI:
  case R_PPC64_TPREL16_DS:
  case R_PPC64_TPREL16_LO_DS:
  case R_PPC64_TPREL16_HIGHER:
  case R_PPC64_TPREL16_HIGHERA:
  case R_PPC64_TPREL16_HIGHEST:
  case R_PPC64_TPREL16_HIGHESTA:
  case R_PPC64_TPREL34:
    return R_TPREL;
  case R_PPC64_DTPREL16:
  case R_PPC64_DTPREL16_DS:
  case R_PPC64_DTPREL16_HA:
  case R_PPC64_DTPREL16_HI:
  case R_PPC64_DTPREL16_HIGHER:
  case R_PPC64_DTPREL16_HIGHERA:
  case R_PPC64_DTPREL16_HIGHEST:
  case R_PPC64_DTPREL16_HIGHESTA:
  case R_PPC64_DTPREL16_LO:
  case R_PPC64_DTPREL16_LO_DS:
  case R_PPC64_DTPREL64:
  case R_PPC64_DTPREL34:
    return R_DTPREL;
  case R_PPC64_TLSGD:
    return R_TLSDESC_CALL;
  case R_PPC64_TLSLD:
    return R_TLSLD_HINT;
  case R_PPC64_TLS:
    return R_TLSIE_HINT;
  default:
    Err(ctx) << getErrorLoc(ctx, loc) << "unknown relocation (" << type.v
             << ") against symbol " << &s;
    return R_NONE;
  }
}

RelType PPC64::getDynRel(RelType type) const {
  if (type == R_PPC64_ADDR64 || type == R_PPC64_TOC)
    return R_PPC64_ADDR64;
  return R_PPC64_NONE;
}

int64_t PPC64::getImplicitAddend(const uint8_t *buf, RelType type) const {
  switch (type) {
  case R_PPC64_NONE:
  case R_PPC64_GLOB_DAT:
  case R_PPC64_JMP_SLOT:
    return 0;
  case R_PPC64_REL32:
    return SignExtend64<32>(read32(ctx, buf));
  case R_PPC64_ADDR64:
  case R_PPC64_REL64:
  case R_PPC64_RELATIVE:
  case R_PPC64_IRELATIVE:
  case R_PPC64_DTPMOD64:
  case R_PPC64_DTPREL64:
  case R_PPC64_TPREL64:
    return read64(ctx, buf);
  default:
    InternalErr(ctx, buf) << "cannot read addend for relocation " << type;
    return 0;
  }
}

void PPC64::writeGotHeader(uint8_t *buf) const {
  write64(ctx, buf, getPPC64TocBase(ctx));
}

void PPC64::writePltHeader(uint8_t *buf) const {
  // The generic resolver stub goes first.
  write32(ctx, buf + 0, 0x7c0802a6);  // mflr r0
  write32(ctx, buf + 4, 0x429f0005);  // bcl  20,4*cr7+so,8 <_glink+0x8>
  write32(ctx, buf + 8, 0x7d6802a6);  // mflr r11
  write32(ctx, buf + 12, 0x7c0803a6); // mtlr r0
  write32(ctx, buf + 16, 0x7d8b6050); // subf r12, r11, r12
  write32(ctx, buf + 20, 0x380cffcc); // subi r0,r12,52
  write32(ctx, buf + 24, 0x7800f082); // srdi r0,r0,62,2
  write32(ctx, buf + 28, 0xe98b002c); // ld   r12,44(r11)
  write32(ctx, buf + 32, 0x7d6c5a14); // add  r11,r12,r11
  write32(ctx, buf + 36, 0xe98b0000); // ld   r12,0(r11)
  write32(ctx, buf + 40, 0xe96b0008); // ld   r11,8(r11)
  write32(ctx, buf + 44, 0x7d8903a6); // mtctr   r12
  write32(ctx, buf + 48, 0x4e800420); // bctr

  // The 'bcl' instruction will set the link register to the address of the
  // following instruction ('mflr r11'). Here we store the offset from that
  // instruction  to the first entry in the GotPlt section.
  int64_t gotPltOffset = ctx.in.gotPlt->getVA() - (ctx.in.plt->getVA() + 8);
  write64(ctx, buf + 52, gotPltOffset);
}

void PPC64::writePlt(uint8_t *buf, const Symbol &sym,
                     uint64_t /*pltEntryAddr*/) const {
  int32_t offset = pltHeaderSize + sym.getPltIdx(ctx) * pltEntrySize;
  // bl __glink_PLTresolve
  write32(ctx, buf, 0x48000000 | ((-offset) & 0x03fffffc));
}

void PPC64::writeIplt(uint8_t *buf, const Symbol &sym,
                      uint64_t /*pltEntryAddr*/) const {
  writePPC64LoadAndBranch(ctx, buf,
                          sym.getGotPltVA(ctx) - getPPC64TocBase(ctx));
}

static std::pair<RelType, uint64_t> toAddr16Rel(RelType type, uint64_t val) {
  // Relocations relative to the toc-base need to be adjusted by the Toc offset.
  uint64_t tocBiasedVal = val - ppc64TocOffset;
  // Relocations relative to dtv[dtpmod] need to be adjusted by the DTP offset.
  uint64_t dtpBiasedVal = val - dynamicThreadPointerOffset;

  switch (type) {
  // TOC biased relocation.
  case R_PPC64_GOT16:
  case R_PPC64_GOT_TLSGD16:
  case R_PPC64_GOT_TLSLD16:
  case R_PPC64_TOC16:
    return {R_PPC64_ADDR16, tocBiasedVal};
  case R_PPC64_GOT16_DS:
  case R_PPC64_TOC16_DS:
  case R_PPC64_GOT_TPREL16_DS:
  case R_PPC64_GOT_DTPREL16_DS:
    return {R_PPC64_ADDR16_DS, tocBiasedVal};
  case R_PPC64_GOT16_HA:
  case R_PPC64_GOT_TLSGD16_HA:
  case R_PPC64_GOT_TLSLD16_HA:
  case R_PPC64_GOT_TPREL16_HA:
  case R_PPC64_GOT_DTPREL16_HA:
  case R_PPC64_TOC16_HA:
    return {R_PPC64_ADDR16_HA, tocBiasedVal};
  case R_PPC64_GOT16_HI:
  case R_PPC64_GOT_TLSGD16_HI:
  case R_PPC64_GOT_TLSLD16_HI:
  case R_PPC64_GOT_TPREL16_HI:
  case R_PPC64_GOT_DTPREL16_HI:
  case R_PPC64_TOC16_HI:
    return {R_PPC64_ADDR16_HI, tocBiasedVal};
  case R_PPC64_GOT16_LO:
  case R_PPC64_GOT_TLSGD16_LO:
  case R_PPC64_GOT_TLSLD16_LO:
  case R_PPC64_TOC16_LO:
    return {R_PPC64_ADDR16_LO, tocBiasedVal};
  case R_PPC64_GOT16_LO_DS:
  case R_PPC64_TOC16_LO_DS:
  case R_PPC64_GOT_TPREL16_LO_DS:
  case R_PPC64_GOT_DTPREL16_LO_DS:
    return {R_PPC64_ADDR16_LO_DS, tocBiasedVal};

  // Dynamic Thread pointer biased relocation types.
  case R_PPC64_DTPREL16:
    return {R_PPC64_ADDR16, dtpBiasedVal};
  case R_PPC64_DTPREL16_DS:
    return {R_PPC64_ADDR16_DS, dtpBiasedVal};
  case R_PPC64_DTPREL16_HA:
    return {R_PPC64_ADDR16_HA, dtpBiasedVal};
  case R_PPC64_DTPREL16_HI:
    return {R_PPC64_ADDR16_HI, dtpBiasedVal};
  case R_PPC64_DTPREL16_HIGHER:
    return {R_PPC64_ADDR16_HIGHER, dtpBiasedVal};
  case R_PPC64_DTPREL16_HIGHERA:
    return {R_PPC64_ADDR16_HIGHERA, dtpBiasedVal};
  case R_PPC64_DTPREL16_HIGHEST:
    return {R_PPC64_ADDR16_HIGHEST, dtpBiasedVal};
  case R_PPC64_DTPREL16_HIGHESTA:
    return {R_PPC64_ADDR16_HIGHESTA, dtpBiasedVal};
  case R_PPC64_DTPREL16_LO:
    return {R_PPC64_ADDR16_LO, dtpBiasedVal};
  case R_PPC64_DTPREL16_LO_DS:
    return {R_PPC64_ADDR16_LO_DS, dtpBiasedVal};
  case R_PPC64_DTPREL64:
    return {R_PPC64_ADDR64, dtpBiasedVal};

  default:
    return {type, val};
  }
}

static bool isTocOptType(RelType type) {
  switch (type) {
  case R_PPC64_GOT16_HA:
  case R_PPC64_GOT16_LO_DS:
  case R_PPC64_TOC16_HA:
  case R_PPC64_TOC16_LO_DS:
  case R_PPC64_TOC16_LO:
    return true;
  default:
    return false;
  }
}

// R_PPC64_TLSGD/R_PPC64_TLSLD is required to mark `bl __tls_get_addr` for
// General Dynamic/Local Dynamic code sequences. If a GD/LD GOT relocation is
// found but no R_PPC64_TLSGD/R_PPC64_TLSLD is seen, we assume that the
// instructions are generated by very old IBM XL compilers. Work around the
// issue by disabling GD/LD to IE/LE relaxation.
template <class RelTy>
static void checkPPC64TLSRelax(InputSectionBase &sec, Relocs<RelTy> rels) {
  // Skip if sec is synthetic (sec.file is null) or if sec has been marked.
  if (!sec.file || sec.file->ppc64DisableTLSRelax)
    return;
  bool hasGDLD = false;
  for (const RelTy &rel : rels) {
    RelType type = rel.getType(false);
    switch (type) {
    case R_PPC64_TLSGD:
    case R_PPC64_TLSLD:
      return; // Found a marker
    case R_PPC64_GOT_TLSGD16:
    case R_PPC64_GOT_TLSGD16_HA:
    case R_PPC64_GOT_TLSGD16_HI:
    case R_PPC64_GOT_TLSGD16_LO:
    case R_PPC64_GOT_TLSLD16:
    case R_PPC64_GOT_TLSLD16_HA:
    case R_PPC64_GOT_TLSLD16_HI:
    case R_PPC64_GOT_TLSLD16_LO:
      hasGDLD = true;
      break;
    }
  }
  if (hasGDLD) {
    sec.file->ppc64DisableTLSRelax = true;
    Warn(sec.file->ctx)
        << sec.file
        << ": disable TLS relaxation due to R_PPC64_GOT_TLS* relocations "
           "without "
           "R_PPC64_TLSGD/R_PPC64_TLSLD relocations";
  }
}

template <class ELFT, class RelTy>
void PPC64::scanSectionImpl(InputSectionBase &sec, Relocs<RelTy> rels) {
  RelocScan rs(ctx, &sec);
  sec.relocations.reserve(rels.size());
  checkPPC64TLSRelax<RelTy>(sec, rels);
  for (auto it = rels.begin(); it != rels.end(); ++it) {
    const RelTy &rel = *it;
    uint64_t offset = rel.r_offset;
    uint32_t symIdx = rel.getSymbol(false);
    Symbol &sym = sec.getFile<ELFT>()->getSymbol(symIdx);
    RelType type = rel.getType(false);
    RelExpr expr =
        ctx.target->getRelExpr(type, sym, sec.content().data() + offset);
    if (expr == R_NONE)
      continue;
    if (sym.isUndefined() && symIdx != 0 &&
        rs.maybeReportUndefined(cast<Undefined>(sym), offset))
      continue;

    auto addend = getAddend<ELFT>(rel);
    if (ctx.arg.isPic && type == R_PPC64_TOC)
      addend += getPPC64TocBase(ctx);

    // We can separate the small code model relocations into 2 categories:
    // 1) Those that access the compiler generated .toc sections.
    // 2) Those that access the linker allocated got entries.
    // lld allocates got entries to symbols on demand. Since we don't try to
    // sort the got entries in any way, we don't have to track which objects
    // have got-based small code model relocs. The .toc sections get placed
    // after the end of the linker allocated .got section and we do sort those
    // so sections addressed with small code model relocations come first.
    if (type == R_PPC64_TOC16 || type == R_PPC64_TOC16_DS)
      sec.file->ppc64SmallCodeModelTocRelocs = true;

    // Record the TOC entry (.toc + addend) as not relaxable. See the comment in
    // PPC64::relocateAlloc().
    if (type == R_PPC64_TOC16_LO && sym.isSection() && isa<Defined>(sym) &&
        cast<Defined>(sym).section->name == ".toc")
      ctx.ppc64noTocRelax.insert({&sym, addend});

    if ((type == R_PPC64_TLSGD && expr == R_TLSDESC_CALL) ||
        (type == R_PPC64_TLSLD && expr == R_TLSLD_HINT)) {
      auto it1 = it;
      ++it1;
      if (it1 == rels.end()) {
        auto diag = Err(ctx);
        diag << "R_PPC64_TLSGD/R_PPC64_TLSLD may not be the last "
                "relocation";
        printLocation(diag, sec, sym, offset);
        continue;
      }

      // Offset the 4-byte aligned R_PPC64_TLSGD by one byte in the NOTOC
      // case, so we can discern it later from the toc-case.
      if (it1->getType(/*isMips64EL=*/false) == R_PPC64_REL24_NOTOC)
        ++offset;
    }

    if (oneof<R_GOTREL, RE_PPC64_TOCBASE, RE_PPC64_RELAX_TOC>(expr))
      ctx.in.got->hasGotOffRel.store(true, std::memory_order_relaxed);

    if (sym.isTls()) {
      if (unsigned processed =
              rs.handleTlsRelocation(expr, type, offset, sym, addend)) {
        it += processed - 1;
        continue;
      }
    }
    rs.process(expr, type, offset, sym, addend);
  }
}

template <class ELFT> void PPC64::scanSection1(InputSectionBase &sec) {
  auto relocs = sec.template relsOrRelas<ELFT>();
  if (relocs.areRelocsCrel())
    scanSectionImpl<ELFT>(sec, relocs.crels);
  else
    scanSectionImpl<ELFT>(sec, relocs.relas);
}

void PPC64::scanSection(InputSectionBase &sec) {
  if (ctx.arg.isLE)
    scanSection1<ELF64LE>(sec);
  else
    scanSection1<ELF64BE>(sec);
}

void PPC64::relocate(uint8_t *loc, const Relocation &rel, uint64_t val) const {
  RelType type = rel.type;
  bool shouldTocOptimize =  isTocOptType(type);
  // For dynamic thread pointer relative, toc-relative, and got-indirect
  // relocations, proceed in terms of the corresponding ADDR16 relocation type.
  std::tie(type, val) = toAddr16Rel(type, val);

  switch (type) {
  case R_PPC64_ADDR14: {
    checkAlignment(ctx, loc, val, 4, rel);
    // Preserve the AA/LK bits in the branch instruction
    uint8_t aalk = loc[3];
    write16(ctx, loc + 2, (aalk & 3) | (val & 0xfffc));
    break;
  }
  case R_PPC64_ADDR16:
    checkIntUInt(ctx, loc, val, 16, rel);
    write16(ctx, loc, val);
    break;
  case R_PPC64_ADDR32:
    checkIntUInt(ctx, loc, val, 32, rel);
    write32(ctx, loc, val);
    break;
  case R_PPC64_ADDR16_DS:
  case R_PPC64_TPREL16_DS: {
    checkInt(ctx, loc, val, 16, rel);
    // DQ-form instructions use bits 28-31 as part of the instruction encoding
    // DS-form instructions only use bits 30-31.
    uint16_t mask = isDQFormInstruction(readFromHalf16(ctx, loc)) ? 0xf : 0x3;
    checkAlignment(ctx, loc, lo(val), mask + 1, rel);
    write16(ctx, loc, (read16(ctx, loc) & mask) | lo(val));
  } break;
  case R_PPC64_ADDR16_HA:
  case R_PPC64_REL16_HA:
  case R_PPC64_TPREL16_HA:
    if (ctx.arg.tocOptimize && shouldTocOptimize && ha(val) == 0)
      writeFromHalf16(ctx, loc, NOP);
    else {
      checkInt(ctx, loc, val + 0x8000, 32, rel);
      write16(ctx, loc, ha(val));
    }
    break;
  case R_PPC64_ADDR16_HI:
  case R_PPC64_REL16_HI:
  case R_PPC64_TPREL16_HI:
    checkInt(ctx, loc, val, 32, rel);
    write16(ctx, loc, hi(val));
    break;
  case R_PPC64_ADDR16_HIGH:
    write16(ctx, loc, hi(val));
    break;
  case R_PPC64_ADDR16_HIGHER:
  case R_PPC64_TPREL16_HIGHER:
    write16(ctx, loc, higher(val));
    break;
  case R_PPC64_ADDR16_HIGHERA:
  case R_PPC64_TPREL16_HIGHERA:
    write16(ctx, loc, highera(val));
    break;
  case R_PPC64_ADDR16_HIGHEST:
  case R_PPC64_TPREL16_HIGHEST:
    write16(ctx, loc, highest(val));
    break;
  case R_PPC64_ADDR16_HIGHESTA:
  case R_PPC64_TPREL16_HIGHESTA:
    write16(ctx, loc, highesta(val));
    break;
  case R_PPC64_ADDR16_LO:
  case R_PPC64_REL16_LO:
  case R_PPC64_TPREL16_LO:
    // When the high-adjusted part of a toc relocation evaluates to 0, it is
    // changed into a nop. The lo part then needs to be updated to use the
    // toc-pointer register r2, as the base register.
    if (ctx.arg.tocOptimize && shouldTocOptimize && ha(val) == 0) {
      uint32_t insn = readFromHalf16(ctx, loc);
      if (isInstructionUpdateForm(insn))
        Err(ctx) << getErrorLoc(ctx, loc)
                 << "can't toc-optimize an update instruction: 0x"
                 << utohexstr(insn, true);
      writeFromHalf16(ctx, loc, (insn & 0xffe00000) | 0x00020000 | lo(val));
    } else {
      write16(ctx, loc, lo(val));
    }
    break;
  case R_PPC64_ADDR16_LO_DS:
  case R_PPC64_TPREL16_LO_DS: {
    // DQ-form instructions use bits 28-31 as part of the instruction encoding
    // DS-form instructions only use bits 30-31.
    uint32_t insn = readFromHalf16(ctx, loc);
    uint16_t mask = isDQFormInstruction(insn) ? 0xf : 0x3;
    checkAlignment(ctx, loc, lo(val), mask + 1, rel);
    if (ctx.arg.tocOptimize && shouldTocOptimize && ha(val) == 0) {
      // When the high-adjusted part of a toc relocation evaluates to 0, it is
      // changed into a nop. The lo part then needs to be updated to use the toc
      // pointer register r2, as the base register.
      if (isInstructionUpdateForm(insn))
        Err(ctx) << getErrorLoc(ctx, loc)
                 << "can't toc-optimize an update instruction: 0x"
                 << utohexstr(insn, true);
      insn &= 0xffe00000 | mask;
      writeFromHalf16(ctx, loc, insn | 0x00020000 | lo(val));
    } else {
      write16(ctx, loc, (read16(ctx, loc) & mask) | lo(val));
    }
  } break;
  case R_PPC64_TPREL16:
    checkInt(ctx, loc, val, 16, rel);
    write16(ctx, loc, val);
    break;
  case R_PPC64_REL32:
    checkInt(ctx, loc, val, 32, rel);
    write32(ctx, loc, val);
    break;
  case R_PPC64_ADDR64:
  case R_PPC64_REL64:
  case R_PPC64_TOC:
    write64(ctx, loc, val);
    break;
  case R_PPC64_REL14: {
    uint32_t mask = 0x0000FFFC;
    checkInt(ctx, loc, val, 16, rel);
    checkAlignment(ctx, loc, val, 4, rel);
    write32(ctx, loc, (read32(ctx, loc) & ~mask) | (val & mask));
    break;
  }
  case R_PPC64_REL24:
  case R_PPC64_REL24_NOTOC: {
    uint32_t mask = 0x03FFFFFC;
    checkInt(ctx, loc, val, 26, rel);
    checkAlignment(ctx, loc, val, 4, rel);
    write32(ctx, loc, (read32(ctx, loc) & ~mask) | (val & mask));
    break;
  }
  case R_PPC64_DTPREL64:
    write64(ctx, loc, val - dynamicThreadPointerOffset);
    break;
  case R_PPC64_DTPREL34:
    // The Dynamic Thread Vector actually points 0x8000 bytes past the start
    // of the TLS block. Therefore, in the case of R_PPC64_DTPREL34 we first
    // need to subtract that value then fallthrough to the general case.
    val -= dynamicThreadPointerOffset;
    [[fallthrough]];
  case R_PPC64_PCREL34:
  case R_PPC64_GOT_PCREL34:
  case R_PPC64_GOT_TLSGD_PCREL34:
  case R_PPC64_GOT_TLSLD_PCREL34:
  case R_PPC64_GOT_TPREL_PCREL34:
  case R_PPC64_TPREL34: {
    const uint64_t si0Mask = 0x00000003ffff0000;
    const uint64_t si1Mask = 0x000000000000ffff;
    const uint64_t fullMask = 0x0003ffff0000ffff;
    checkInt(ctx, loc, val, 34, rel);

    uint64_t instr = readPrefixedInst(ctx, loc) & ~fullMask;
    writePrefixedInst(ctx, loc,
                      instr | ((val & si0Mask) << 16) | (val & si1Mask));
    break;
  }
  // If we encounter a PCREL_OPT relocation that we won't optimize.
  case R_PPC64_PCREL_OPT:
    break;
  default:
    llvm_unreachable("unknown relocation");
  }
}

bool PPC64::needsThunk(RelExpr expr, RelType type, const InputFile *file,
                       uint64_t branchAddr, const Symbol &s, int64_t a) const {
  if (type != R_PPC64_REL14 && type != R_PPC64_REL24 &&
      type != R_PPC64_REL24_NOTOC)
    return false;

  // If a function is in the Plt it needs to be called with a call-stub.
  if (s.isInPlt(ctx))
    return true;

  // This check looks at the st_other bits of the callee with relocation
  // R_PPC64_REL14 or R_PPC64_REL24. If the value is 1, then the callee
  // clobbers the TOC and we need an R2 save stub.
  if (type != R_PPC64_REL24_NOTOC && (s.stOther >> 5) == 1)
    return true;

  if (type == R_PPC64_REL24_NOTOC && (s.stOther >> 5) > 1)
    return true;

  // An undefined weak symbol not in a PLT does not need a thunk. If it is
  // hidden, its binding has been converted to local, so we just check
  // isUndefined() here. A undefined non-weak symbol has been errored.
  if (s.isUndefined())
    return false;

  // If the offset exceeds the range of the branch type then it will need
  // a range-extending thunk.
  // See the comment in getRelocTargetVA() about RE_PPC64_CALL.
  return !inBranchRange(
      type, branchAddr,
      s.getVA(ctx, a) + getPPC64GlobalEntryToLocalEntryOffset(ctx, s.stOther));
}

uint32_t PPC64::getThunkSectionSpacing() const {
  // See comment in Arch/ARM.cpp for a more detailed explanation of
  // getThunkSectionSpacing(). For PPC64 we pick the constant here based on
  // R_PPC64_REL24, which is used by unconditional branch instructions.
  // 0x2000000 = (1 << 24-1) * 4
  return 0x2000000;
}

bool PPC64::inBranchRange(RelType type, uint64_t src, uint64_t dst) const {
  int64_t offset = dst - src;
  if (type == R_PPC64_REL14)
    return isInt<16>(offset);
  if (type == R_PPC64_REL24 || type == R_PPC64_REL24_NOTOC)
    return isInt<26>(offset);
  llvm_unreachable("unsupported relocation type used in branch");
}

RelExpr PPC64::adjustTlsExpr(RelType type, RelExpr expr) const {
  if (type != R_PPC64_GOT_TLSGD_PCREL34 && expr == R_RELAX_TLS_GD_TO_IE)
    return R_RELAX_TLS_GD_TO_IE_GOT_OFF;
  if (expr == R_RELAX_TLS_LD_TO_LE)
    return R_RELAX_TLS_LD_TO_LE_ABS;
  return expr;
}

RelExpr PPC64::adjustGotPcExpr(RelType type, int64_t addend,
                               const uint8_t *loc) const {
  if ((type == R_PPC64_GOT_PCREL34 || type == R_PPC64_PCREL_OPT) &&
      ctx.arg.pcRelOptimize) {
    // It only makes sense to optimize pld since paddi means that the address
    // of the object in the GOT is required rather than the object itself.
    if ((readPrefixedInst(ctx, loc) & 0xfc000000) == 0xe4000000)
      return RE_PPC64_RELAX_GOT_PC;
  }
  return R_GOT_PC;
}

// Reference: 3.7.4.1 of the 64-bit ELF V2 abi supplement.
// The general dynamic code sequence for a global `x` uses 4 instructions.
// Instruction                    Relocation                Symbol
// addis r3, r2, x@got@tlsgd@ha   R_PPC64_GOT_TLSGD16_HA      x
// addi  r3, r3, x@got@tlsgd@l    R_PPC64_GOT_TLSGD16_LO      x
// bl __tls_get_addr(x@tlsgd)     R_PPC64_TLSGD               x
//                                R_PPC64_REL24               __tls_get_addr
// nop                            None                       None
//
// Relaxing to initial-exec entails:
// 1) Convert the addis/addi pair that builds the address of the tls_index
//    struct for 'x' to an addis/ld pair that loads an offset from a got-entry.
// 2) Convert the call to __tls_get_addr to a nop.
// 3) Convert the nop following the call to an add of the loaded offset to the
//    thread pointer.
// Since the nop must directly follow the call, the R_PPC64_TLSGD relocation is
// used as the relaxation hint for both steps 2 and 3.
void PPC64::relaxTlsGdToIe(uint8_t *loc, const Relocation &rel,
                           uint64_t val) const {
  switch (rel.type) {
  case R_PPC64_GOT_TLSGD16_HA:
    // This is relaxed from addis rT, r2, sym@got@tlsgd@ha to
    //                      addis rT, r2, sym@got@tprel@ha.
    relocateNoSym(loc, R_PPC64_GOT_TPREL16_HA, val);
    return;
  case R_PPC64_GOT_TLSGD16:
  case R_PPC64_GOT_TLSGD16_LO: {
    // Relax from addi  r3, rA, sym@got@tlsgd@l to
    //            ld r3, sym@got@tprel@l(rA)
    uint32_t ra = (readFromHalf16(ctx, loc) & (0x1f << 16));
    writeFromHalf16(ctx, loc, 0xe8600000 | ra);
    relocateNoSym(loc, R_PPC64_GOT_TPREL16_LO_DS, val);
    return;
  }
  case R_PPC64_GOT_TLSGD_PCREL34: {
    // Relax from paddi r3, 0, sym@got@tlsgd@pcrel, 1 to
    //            pld r3, sym@got@tprel@pcrel
    writePrefixedInst(ctx, loc, 0x04100000e4600000);
    relocateNoSym(loc, R_PPC64_GOT_TPREL_PCREL34, val);
    return;
  }
  case R_PPC64_TLSGD: {
    // PC Relative Relaxation:
    // Relax from bl __tls_get_addr@notoc(x@tlsgd) to
    //            nop
    // TOC Relaxation:
    // Relax from bl __tls_get_addr(x@tlsgd)
    //            nop
    // to
    //            nop
    //            add r3, r3, r13
    const uintptr_t locAsInt = reinterpret_cast<uintptr_t>(loc);
    if (locAsInt % 4 == 0) {
      write32(ctx, loc, NOP);            // bl __tls_get_addr(sym@tlsgd) --> nop
      write32(ctx, loc + 4, 0x7c636a14); // nop --> add r3, r3, r13
    } else if (locAsInt % 4 == 1) {
      // bl __tls_get_addr(sym@tlsgd) --> add r3, r3, r13
      write32(ctx, loc - 1, 0x7c636a14);
    } else {
      Err(ctx) << "R_PPC64_TLSGD has unexpected byte alignment";
    }
    return;
  }
  default:
    llvm_unreachable("unsupported relocation for TLS GD to IE relaxation");
  }
}

void PPC64::relocateAlloc(InputSection &sec, uint8_t *buf) const {
  uint64_t secAddr = sec.getOutputSection()->addr + sec.outSecOff;
  uint64_t lastPPCRelaxedRelocOff = -1;
  for (const Relocation &rel : sec.relocs()) {
    uint8_t *loc = buf + rel.offset;
    const uint64_t val = sec.getRelocTargetVA(ctx, rel, secAddr + rel.offset);
    switch (rel.expr) {
    case RE_PPC64_RELAX_GOT_PC: {
      // The R_PPC64_PCREL_OPT relocation must appear immediately after
      // R_PPC64_GOT_PCREL34 in the relocations table at the same offset.
      // We can only relax R_PPC64_PCREL_OPT if we have also relaxed
      // the associated R_PPC64_GOT_PCREL34 since only the latter has an
      // associated symbol. So save the offset when relaxing R_PPC64_GOT_PCREL34
      // and only relax the other if the saved offset matches.
      if (rel.type == R_PPC64_GOT_PCREL34)
        lastPPCRelaxedRelocOff = rel.offset;
      if (rel.type == R_PPC64_PCREL_OPT && rel.offset != lastPPCRelaxedRelocOff)
        break;
      relaxGot(loc, rel, val);
      break;
    }
    case RE_PPC64_RELAX_TOC:
      // rel.sym refers to the STT_SECTION symbol associated to the .toc input
      // section. If an R_PPC64_TOC16_LO (.toc + addend) references the TOC
      // entry, there may be R_PPC64_TOC16_HA not paired with
      // R_PPC64_TOC16_LO_DS. Don't relax. This loses some relaxation
      // opportunities but is safe.
      if (ctx.ppc64noTocRelax.count({rel.sym, rel.addend}) ||
          !tryRelaxPPC64TocIndirection(ctx, rel, loc))
        relocate(loc, rel, val);
      break;
    case RE_PPC64_CALL:
      // If this is a call to __tls_get_addr, it may be part of a TLS
      // sequence that has been relaxed and turned into a nop. In this
      // case, we don't want to handle it as a call.
      if (read32(ctx, loc) == 0x60000000) // nop
        break;

      // Patch a nop (0x60000000) to a ld.
      if (rel.sym->needsTocRestore()) {
        // gcc/gfortran 5.4, 6.3 and earlier versions do not add nop for
        // recursive calls even if the function is preemptible. This is not
        // wrong in the common case where the function is not preempted at
        // runtime. Just ignore.
        if ((rel.offset + 8 > sec.content().size() ||
             read32(ctx, loc + 4) != 0x60000000) &&
            rel.sym->file != sec.file) {
          // Use substr(6) to remove the "__plt_" prefix.
          Err(ctx) << getErrorLoc(ctx, loc) << "call to "
                   << toStr(ctx, *rel.sym).substr(6)
                   << " lacks nop, can't restore toc";
          break;
        }
        write32(ctx, loc + 4, 0xe8410018); // ld %r2, 24(%r1)
      }
      relocate(loc, rel, val);
      break;
    case R_RELAX_TLS_GD_TO_IE:
    case R_RELAX_TLS_GD_TO_IE_GOT_OFF:
      relaxTlsGdToIe(loc, rel, val);
      break;
    case R_RELAX_TLS_GD_TO_LE:
      relaxTlsGdToLe(loc, rel, val);
      break;
    case R_RELAX_TLS_LD_TO_LE_ABS:
      relaxTlsLdToLe(loc, rel, val);
      break;
    case R_RELAX_TLS_IE_TO_LE:
      relaxTlsIeToLe(loc, rel, val);
      break;
    default:
      relocate(loc, rel, val);
      break;
    }
  }
}

// The prologue for a split-stack function is expected to look roughly
// like this:
//    .Lglobal_entry_point:
//      # TOC pointer initialization.
//      ...
//    .Llocal_entry_point:
//      # load the __private_ss member of the threads tcbhead.
//      ld r0,-0x7000-64(r13)
//      # subtract the functions stack size from the stack pointer.
//      addis r12, r1, ha(-stack-frame size)
//      addi  r12, r12, l(-stack-frame size)
//      # compare needed to actual and branch to allocate_more_stack if more
//      # space is needed, otherwise fallthrough to 'normal' function body.
//      cmpld cr7,r12,r0
//      blt- cr7, .Lallocate_more_stack
//
// -) The allocate_more_stack block might be placed after the split-stack
//    prologue and the `blt-` replaced with a `bge+ .Lnormal_func_body`
//    instead.
// -) If either the addis or addi is not needed due to the stack size being
//    smaller then 32K or a multiple of 64K they will be replaced with a nop,
//    but there will always be 2 instructions the linker can overwrite for the
//    adjusted stack size.
//
// The linkers job here is to increase the stack size used in the addis/addi
// pair by split-stack-size-adjust.
// addis r12, r1, ha(-stack-frame size - split-stack-adjust-size)
// addi  r12, r12, l(-stack-frame size - split-stack-adjust-size)
bool PPC64::adjustPrologueForCrossSplitStack(uint8_t *loc, uint8_t *end,
                                             uint8_t stOther) const {
  // If the caller has a global entry point adjust the buffer past it. The start
  // of the split-stack prologue will be at the local entry point.
  loc += getPPC64GlobalEntryToLocalEntryOffset(ctx, stOther);

  // At the very least we expect to see a load of some split-stack data from the
  // tcb, and 2 instructions that calculate the ending stack address this
  // function will require. If there is not enough room for at least 3
  // instructions it can't be a split-stack prologue.
  if (loc + 12 >= end)
    return false;

  // First instruction must be `ld r0, -0x7000-64(r13)`
  if (read32(ctx, loc) != 0xe80d8fc0)
    return false;

  int16_t hiImm = 0;
  int16_t loImm = 0;
  // First instruction can be either an addis if the frame size is larger then
  // 32K, or an addi if the size is less then 32K.
  int32_t firstInstr = read32(ctx, loc + 4);
  if (getPrimaryOpCode(firstInstr) == 15) {
    hiImm = firstInstr & 0xFFFF;
  } else if (getPrimaryOpCode(firstInstr) == 14) {
    loImm = firstInstr & 0xFFFF;
  } else {
    return false;
  }

  // Second instruction is either an addi or a nop. If the first instruction was
  // an addi then LoImm is set and the second instruction must be a nop.
  uint32_t secondInstr = read32(ctx, loc + 8);
  if (!loImm && getPrimaryOpCode(secondInstr) == 14) {
    loImm = secondInstr & 0xFFFF;
  } else if (secondInstr != NOP) {
    return false;
  }

  // The register operands of the first instruction should be the stack-pointer
  // (r1) as the input (RA) and r12 as the output (RT). If the second
  // instruction is not a nop, then it should use r12 as both input and output.
  auto checkRegOperands = [](uint32_t instr, uint8_t expectedRT,
                             uint8_t expectedRA) {
    return ((instr & 0x3E00000) >> 21 == expectedRT) &&
           ((instr & 0x1F0000) >> 16 == expectedRA);
  };
  if (!checkRegOperands(firstInstr, 12, 1))
    return false;
  if (secondInstr != NOP && !checkRegOperands(secondInstr, 12, 12))
    return false;

  int32_t stackFrameSize = (hiImm * 65536) + loImm;
  // Check that the adjusted size doesn't overflow what we can represent with 2
  // instructions.
  if (stackFrameSize < ctx.arg.splitStackAdjustSize + INT32_MIN) {
    Err(ctx) << getErrorLoc(ctx, loc)
             << "split-stack prologue adjustment overflows";
    return false;
  }

  int32_t adjustedStackFrameSize =
      stackFrameSize - ctx.arg.splitStackAdjustSize;

  loImm = adjustedStackFrameSize & 0xFFFF;
  hiImm = (adjustedStackFrameSize + 0x8000) >> 16;
  if (hiImm) {
    write32(ctx, loc + 4, 0x3d810000 | (uint16_t)hiImm);
    // If the low immediate is zero the second instruction will be a nop.
    secondInstr = loImm ? 0x398C0000 | (uint16_t)loImm : NOP;
    write32(ctx, loc + 8, secondInstr);
  } else {
    // addi r12, r1, imm
    write32(ctx, loc + 4, (0x39810000) | (uint16_t)loImm);
    write32(ctx, loc + 8, NOP);
  }

  return true;
}

void elf::setPPC64TargetInfo(Ctx &ctx) { ctx.target.reset(new PPC64(ctx)); }
