//===- Target.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Machine-specific things, such as applying relocations, creation of
// GOT or PLT entries, etc., are handled in this file.
//
// Refer the ELF spec for the single letter variables, S, A or P, used
// in this file.
//
// Some functions defined in this file has "relaxTls" as part of their names.
// They do peephole optimization for TLS variables by rewriting instructions.
// They are not part of the ABI but optional optimization, so you can skip
// them if you are not interested in how TLS variables are optimized.
// See the following paper for the details.
//
//   Ulrich Drepper, ELF Handling For Thread-Local Storage
//   http://www.akkadia.org/drepper/tls.pdf
//
//===----------------------------------------------------------------------===//

#include "Target.h"
#include "InputFiles.h"
#include "OutputSections.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Object/ELF.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

std::string elf::toStr(Ctx &ctx, RelType type) {
  StringRef s = getELFRelocationTypeName(ctx.arg.emachine, type);
  if (s == "Unknown")
    return ("Unknown (" + Twine(type) + ")").str();
  return std::string(s);
}

const ELFSyncStream &elf::operator<<(const ELFSyncStream &s, RelType type) {
  s << toStr(s.ctx, type);
  return s;
}

void elf::setTarget(Ctx &ctx) {
  switch (ctx.arg.emachine) {
  case EM_386:
  case EM_IAMCU:
    return setX86TargetInfo(ctx);
  case EM_AARCH64:
    return setAArch64TargetInfo(ctx);
  case EM_AMDGPU:
    return setAMDGPUTargetInfo(ctx);
  case EM_ARM:
    return setARMTargetInfo(ctx);
  case EM_AVR:
    return setAVRTargetInfo(ctx);
  case EM_HEXAGON:
    return setHexagonTargetInfo(ctx);
  case EM_LOONGARCH:
    return setLoongArchTargetInfo(ctx);
  case EM_MIPS:
    return setMipsTargetInfo(ctx);
  case EM_MSP430:
    return setMSP430TargetInfo(ctx);
  case EM_PPC:
    return setPPCTargetInfo(ctx);
  case EM_PPC64:
    return setPPC64TargetInfo(ctx);
  case EM_RISCV:
    return setRISCVTargetInfo(ctx);
  case EM_SPARCV9:
    return setSPARCV9TargetInfo(ctx);
  case EM_S390:
    return setSystemZTargetInfo(ctx);
  case EM_X86_64:
    return setX86_64TargetInfo(ctx);
  default:
    Fatal(ctx) << "unsupported e_machine value: " << ctx.arg.emachine;
  }
}

ErrorPlace elf::getErrorPlace(Ctx &ctx, const uint8_t *loc) {
  assert(loc != nullptr);
  for (InputSectionBase *d : ctx.inputSections) {
    auto *isec = dyn_cast<InputSection>(d);
    if (!isec || !isec->getParent() || (isec->type & SHT_NOBITS))
      continue;

    const uint8_t *isecLoc =
        ctx.bufferStart
            ? (ctx.bufferStart + isec->getParent()->offset + isec->outSecOff)
            : isec->contentMaybeDecompress().data();
    if (isecLoc == nullptr) {
      assert(isa<SyntheticSection>(isec) && "No data but not synthetic?");
      continue;
    }
    if (isecLoc <= loc && loc < isecLoc + isec->getSize()) {
      std::string objLoc = isec->getLocation(loc - isecLoc);
      // Return object file location and source file location.
      Undefined dummy(ctx.internalFile, "", STB_LOCAL, 0, 0);
      ELFSyncStream msg(ctx, DiagLevel::None);
      if (isec->file)
        msg << isec->getSrcMsg(dummy, loc - isecLoc);
      return {isec, objLoc + ": ", std::string(msg.str())};
    }
  }
  return {};
}

TargetInfo::~TargetInfo() {}

int64_t TargetInfo::getImplicitAddend(const uint8_t *buf, RelType type) const {
  InternalErr(ctx, buf) << "cannot read addend for relocation " << type;
  return 0;
}

bool TargetInfo::usesOnlyLowPageBits(RelType type) const { return false; }

bool TargetInfo::needsThunk(RelExpr expr, RelType type, const InputFile *file,
                            uint64_t branchAddr, const Symbol &s,
                            int64_t a) const {
  return false;
}

bool TargetInfo::adjustPrologueForCrossSplitStack(uint8_t *loc, uint8_t *end,
                                                  uint8_t stOther) const {
  Err(ctx) << "target doesn't support split stacks";
  return false;
}

bool TargetInfo::inBranchRange(RelType type, uint64_t src, uint64_t dst) const {
  return true;
}

RelExpr TargetInfo::adjustTlsExpr(RelType type, RelExpr expr) const {
  return expr;
}

RelExpr TargetInfo::adjustGotPcExpr(RelType type, int64_t addend,
                                    const uint8_t *data) const {
  return R_GOT_PC;
}

void TargetInfo::relocateAlloc(InputSectionBase &sec, uint8_t *buf) const {
  const unsigned bits = ctx.arg.is64 ? 64 : 32;
  uint64_t secAddr = sec.getOutputSection()->addr;
  if (auto *s = dyn_cast<InputSection>(&sec))
    secAddr += s->outSecOff;
  else if (auto *ehIn = dyn_cast<EhInputSection>(&sec))
    secAddr += ehIn->getParent()->outSecOff;
  for (const Relocation &rel : sec.relocs()) {
    uint8_t *loc = buf + rel.offset;
    const uint64_t val = SignExtend64(
        sec.getRelocTargetVA(ctx, rel, secAddr + rel.offset), bits);
    if (rel.expr != R_RELAX_HINT)
      relocate(loc, rel, val);
  }
}

uint64_t TargetInfo::getImageBase() const {
  // Use --image-base if set. Fall back to the target default if not.
  if (ctx.arg.imageBase)
    return *ctx.arg.imageBase;
  return ctx.arg.isPic ? 0 : defaultImageBase;
}
