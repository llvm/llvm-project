//===- Target.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_TARGET_H
#define LLD_ELF_TARGET_H

#include "Config.h"
#include "InputSection.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include <array>

namespace lld {
namespace elf {
class Defined;
class InputFile;
class Symbol;

std::string toStr(Ctx &, RelType type);

class TargetInfo {
public:
  TargetInfo(Ctx &ctx) : ctx(ctx) {}
  virtual uint32_t calcEFlags() const { return 0; }
  virtual RelExpr getRelExpr(RelType type, const Symbol &s,
                             const uint8_t *loc) const = 0;
  virtual RelType getDynRel(RelType type) const { return 0; }
  virtual void writeGotPltHeader(uint8_t *buf) const {}
  virtual void writeGotHeader(uint8_t *buf) const {}
  virtual void writeGotPlt(uint8_t *buf, const Symbol &s) const {};
  virtual void writeIgotPlt(uint8_t *buf, const Symbol &s) const {}
  virtual int64_t getImplicitAddend(const uint8_t *buf, RelType type) const;
  virtual int getTlsGdRelaxSkip(RelType type) const { return 1; }

  // If lazy binding is supported, the first entry of the PLT has code
  // to call the dynamic linker to resolve PLT entries the first time
  // they are called. This function writes that code.
  virtual void writePltHeader(uint8_t *buf) const {}

  virtual void writePlt(uint8_t *buf, const Symbol &sym,
                        uint64_t pltEntryAddr) const {}
  virtual void writeIplt(uint8_t *buf, const Symbol &sym,
                         uint64_t pltEntryAddr) const {
    // All but PPC32 and PPC64 use the same format for .plt and .iplt entries.
    writePlt(buf, sym, pltEntryAddr);
  }
  virtual void writeIBTPlt(uint8_t *buf, size_t numEntries) const {}
  virtual void addPltHeaderSymbols(InputSection &isec) const {}
  virtual void addPltSymbols(InputSection &isec, uint64_t off) const {}

  // Returns true if a relocation only uses the low bits of a value such that
  // all those bits are in the same page. For example, if the relocation
  // only uses the low 12 bits in a system with 4k pages. If this is true, the
  // bits will always have the same value at runtime and we don't have to emit
  // a dynamic relocation.
  virtual bool usesOnlyLowPageBits(RelType type) const;

  // Decide whether a Thunk is needed for the relocation from File
  // targeting S.
  virtual bool needsThunk(RelExpr expr, RelType relocType,
                          const InputFile *file, uint64_t branchAddr,
                          const Symbol &s, int64_t a) const;

  // On systems with range extensions we place collections of Thunks at
  // regular spacings that enable the majority of branches reach the Thunks.
  // a value of 0 means range extension thunks are not supported.
  virtual uint32_t getThunkSectionSpacing() const { return 0; }

  // The function with a prologue starting at Loc was compiled with
  // -fsplit-stack and it calls a function compiled without. Adjust the prologue
  // to do the right thing. See https://gcc.gnu.org/wiki/SplitStacks.
  // The symbols st_other flags are needed on PowerPC64 for determining the
  // offset to the split-stack prologue.
  virtual bool adjustPrologueForCrossSplitStack(uint8_t *loc, uint8_t *end,
                                                uint8_t stOther) const;

  // Return true if we can reach dst from src with RelType type.
  virtual bool inBranchRange(RelType type, uint64_t src,
                             uint64_t dst) const;

  virtual void relocate(uint8_t *loc, const Relocation &rel,
                        uint64_t val) const = 0;
  void relocateNoSym(uint8_t *loc, RelType type, uint64_t val) const {
    relocate(loc, Relocation{R_NONE, type, 0, 0, nullptr}, val);
  }
  virtual void relocateAlloc(InputSectionBase &sec, uint8_t *buf) const;

  // Do a linker relaxation pass and return true if we changed something.
  virtual bool relaxOnce(int pass) const { return false; }
  virtual bool synthesizeAlign(uint64_t &dot, InputSection *sec) {
    return false;
  }
  // Do finalize relaxation after collecting relaxation infos.
  virtual void finalizeRelax(int passes) const {}

  virtual void applyJumpInstrMod(uint8_t *loc, JumpModType type,
                                 JumpModType val) const {}
  virtual void applyBranchToBranchOpt() const {}

  virtual ~TargetInfo();

  // This deletes a jump insn at the end of the section if it is a fall thru to
  // the next section.  Further, if there is a conditional jump and a direct
  // jump consecutively, it tries to flip the conditional jump to convert the
  // direct jump into a fall thru and delete it.  Returns true if a jump
  // instruction can be deleted.
  virtual bool deleteFallThruJmpInsn(InputSection &is, InputFile *file,
                                     InputSection *nextIS) const {
    return false;
  }

  Ctx &ctx;
  unsigned defaultCommonPageSize = 4096;
  unsigned defaultMaxPageSize = 4096;

  uint64_t getImageBase() const;

  // True if _GLOBAL_OFFSET_TABLE_ is relative to .got.plt, false if .got.
  bool gotBaseSymInGotPlt = false;

  static constexpr RelType noneRel = 0;
  RelType copyRel = 0;
  RelType gotRel = 0;
  RelType pltRel = 0;
  RelType relativeRel = 0;
  RelType iRelativeRel = 0;
  RelType symbolicRel = 0;
  RelType tlsDescRel = 0;
  RelType tlsGotRel = 0;
  RelType tlsModuleIndexRel = 0;
  RelType tlsOffsetRel = 0;
  unsigned gotEntrySize = ctx.arg.wordsize;
  unsigned pltEntrySize = 0;
  unsigned pltHeaderSize = 0;
  unsigned ipltEntrySize = 0;

  // At least on x86_64 positions 1 and 2 are used by the first plt entry
  // to support lazy loading.
  unsigned gotPltHeaderEntriesNum = 3;

  // On PPC ELF V2 abi, the first entry in the .got is the .TOC.
  unsigned gotHeaderEntriesNum = 0;

  // On PPC ELF V2 abi, the dynamic section needs DT_PPC64_OPT (DT_LOPROC + 3)
  // to be set to 0x2 if there can be multiple TOC's. Although we do not emit
  // multiple TOC's, there can be a mix of TOC and NOTOC addressing which
  // is functionally equivalent.
  int ppc64DynamicSectionOpt = 0;

  bool needsThunks = false;

  // A 4-byte field corresponding to one or more trap instructions, used to pad
  // executable OutputSections.
  std::array<uint8_t, 4> trapInstr = {};

  // Stores the NOP instructions of different sizes for the target and is used
  // to pad sections that are relaxed.
  std::optional<std::vector<std::vector<uint8_t>>> nopInstrs;

  // If a target needs to rewrite calls to __morestack to instead call
  // __morestack_non_split when a split-stack enabled caller calls a
  // non-split-stack callee this will return true. Otherwise returns false.
  bool needsMoreStackNonSplit = true;

  virtual RelExpr adjustTlsExpr(RelType type, RelExpr expr) const;
  virtual RelExpr adjustGotPcExpr(RelType type, int64_t addend,
                                  const uint8_t *loc) const;

protected:
  // On FreeBSD x86_64 the first page cannot be mmaped.
  // On Linux this is controlled by vm.mmap_min_addr. At least on some x86_64
  // installs this is set to 65536, so the first 15 pages cannot be used.
  // Given that, the smallest value that can be used in here is 0x10000.
  uint64_t defaultImageBase = 0x10000;
};

void setAArch64TargetInfo(Ctx &);
void setAMDGPUTargetInfo(Ctx &);
void setARMTargetInfo(Ctx &);
void setAVRTargetInfo(Ctx &);
void setHexagonTargetInfo(Ctx &);
void setLoongArchTargetInfo(Ctx &);
void setMSP430TargetInfo(Ctx &);
void setMipsTargetInfo(Ctx &);
void setPPC64TargetInfo(Ctx &);
void setPPCTargetInfo(Ctx &);
void setRISCVTargetInfo(Ctx &);
void setSPARCV9TargetInfo(Ctx &);
void setSystemZTargetInfo(Ctx &);
void setX86TargetInfo(Ctx &);
void setX86_64TargetInfo(Ctx &);

struct ErrorPlace {
  InputSectionBase *isec;
  std::string loc;
  std::string srcLoc;
};

// Returns input section and corresponding source string for the given location.
ErrorPlace getErrorPlace(Ctx &ctx, const uint8_t *loc);

static inline std::string getErrorLoc(Ctx &ctx, const uint8_t *loc) {
  return getErrorPlace(ctx, loc).loc;
}

void processArmCmseSymbols(Ctx &);

template <class ELFT> uint32_t calcMipsEFlags(Ctx &);
uint8_t getMipsFpAbiFlag(Ctx &, InputFile *file, uint8_t oldFlag,
                         uint8_t newFlag);
uint64_t getMipsPageAddr(uint64_t addr);
bool isMipsN32Abi(Ctx &, const InputFile &f);
bool isMicroMips(Ctx &);
bool isMipsR6(Ctx &);

void writePPC32GlinkSection(Ctx &, uint8_t *buf, size_t numEntries);

unsigned getPPCDFormOp(unsigned secondaryOp);
unsigned getPPCDSFormOp(unsigned secondaryOp);

// In the PowerPC64 Elf V2 abi a function can have 2 entry points.  The first
// is a global entry point (GEP) which typically is used to initialize the TOC
// pointer in general purpose register 2.  The second is a local entry
// point (LEP) which bypasses the TOC pointer initialization code. The
// offset between GEP and LEP is encoded in a function's st_other flags.
// This function will return the offset (in bytes) from the global entry-point
// to the local entry-point.
unsigned getPPC64GlobalEntryToLocalEntryOffset(Ctx &, uint8_t stOther);

// Write a prefixed instruction, which is a 4-byte prefix followed by a 4-byte
// instruction (regardless of endianness). Therefore, the prefix is always in
// lower memory than the instruction.
void writePrefixedInst(Ctx &, uint8_t *loc, uint64_t insn);

void addPPC64SaveRestore(Ctx &);
uint64_t getPPC64TocBase(Ctx &ctx);
uint64_t getAArch64Page(uint64_t expr);
bool isAArch64BTILandingPad(Ctx &, Symbol &s, int64_t a);
template <typename ELFT> void writeARMCmseImportLib(Ctx &);
uint64_t getLoongArchPageDelta(uint64_t dest, uint64_t pc, RelType type);
void riscvFinalizeRelax(int passes);
void mergeRISCVAttributesSections(Ctx &);
void mergeHexagonAttributesSections(Ctx &);
void addArmInputSectionMappingSymbols(Ctx &);
void addArmSyntheticSectionMappingSymbol(Defined *);
void sortArmMappingSymbols(Ctx &);
void convertArmInstructionstoBE8(Ctx &, InputSection *sec, uint8_t *buf);
void createTaggedSymbols(Ctx &);
void initSymbolAnchors(Ctx &);

void setTarget(Ctx &);

template <class ELFT> bool isMipsPIC(const Defined *sym);

const ELFSyncStream &operator<<(const ELFSyncStream &, RelType);

void reportRangeError(Ctx &, uint8_t *loc, const Relocation &rel,
                      const Twine &v, int64_t min, uint64_t max);
void reportRangeError(Ctx &ctx, uint8_t *loc, int64_t v, int n,
                      const Symbol &sym, const Twine &msg);

// Make sure that V can be represented as an N bit signed integer.
inline void checkInt(Ctx &ctx, uint8_t *loc, int64_t v, int n,
                     const Relocation &rel) {
  if (v != llvm::SignExtend64(v, n))
    reportRangeError(ctx, loc, rel, Twine(v), llvm::minIntN(n),
                     llvm::maxIntN(n));
}

// Make sure that V can be represented as an N bit unsigned integer.
inline void checkUInt(Ctx &ctx, uint8_t *loc, uint64_t v, int n,
                      const Relocation &rel) {
  if ((v >> n) != 0)
    reportRangeError(ctx, loc, rel, Twine(v), 0, llvm::maxUIntN(n));
}

// Make sure that V can be represented as an N bit signed or unsigned integer.
inline void checkIntUInt(Ctx &ctx, uint8_t *loc, uint64_t v, int n,
                         const Relocation &rel) {
  // For the error message we should cast V to a signed integer so that error
  // messages show a small negative value rather than an extremely large one
  if (v != (uint64_t)llvm::SignExtend64(v, n) && (v >> n) != 0)
    reportRangeError(ctx, loc, rel, Twine((int64_t)v), llvm::minIntN(n),
                     llvm::maxUIntN(n));
}

inline void checkAlignment(Ctx &ctx, uint8_t *loc, uint64_t v, int n,
                           const Relocation &rel) {
  if ((v & (n - 1)) != 0)
    Err(ctx) << getErrorLoc(ctx, loc) << "improper alignment for relocation "
             << rel.type << ": 0x" << llvm::utohexstr(v)
             << " is not aligned to " << n << " bytes";
}

// Endianness-aware read/write.
inline uint16_t read16(Ctx &ctx, const void *p) {
  return llvm::support::endian::read16(p, ctx.arg.endianness);
}

inline uint32_t read32(Ctx &ctx, const void *p) {
  return llvm::support::endian::read32(p, ctx.arg.endianness);
}

inline uint64_t read64(Ctx &ctx, const void *p) {
  return llvm::support::endian::read64(p, ctx.arg.endianness);
}

inline void write16(Ctx &ctx, void *p, uint16_t v) {
  llvm::support::endian::write16(p, v, ctx.arg.endianness);
}

inline void write32(Ctx &ctx, void *p, uint32_t v) {
  llvm::support::endian::write32(p, v, ctx.arg.endianness);
}

inline void write64(Ctx &ctx, void *p, uint64_t v) {
  llvm::support::endian::write64(p, v, ctx.arg.endianness);
}

// Overwrite a ULEB128 value and keep the original length.
inline uint64_t overwriteULEB128(uint8_t *bufLoc, uint64_t val) {
  while (*bufLoc & 0x80) {
    *bufLoc++ = 0x80 | (val & 0x7f);
    val >>= 7;
  }
  *bufLoc = val;
  return val;
}
} // namespace elf
} // namespace lld

#ifdef __clang__
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#define invokeELFT(f, ...)                                                     \
  do {                                                                         \
    switch (ctx.arg.ekind) {                                                   \
    case lld::elf::ELF32LEKind:                                                \
      f<llvm::object::ELF32LE>(__VA_ARGS__);                                   \
      break;                                                                   \
    case lld::elf::ELF32BEKind:                                                \
      f<llvm::object::ELF32BE>(__VA_ARGS__);                                   \
      break;                                                                   \
    case lld::elf::ELF64LEKind:                                                \
      f<llvm::object::ELF64LE>(__VA_ARGS__);                                   \
      break;                                                                   \
    case lld::elf::ELF64BEKind:                                                \
      f<llvm::object::ELF64BE>(__VA_ARGS__);                                   \
      break;                                                                   \
    default:                                                                   \
      llvm_unreachable("unknown ctx.arg.ekind");                               \
    }                                                                          \
  } while (0)

#endif
