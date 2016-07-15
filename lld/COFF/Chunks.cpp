//===- Chunks.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Chunks.h"
#include "Error.h"
#include "InputFiles.h"
#include "Symbols.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::COFF;
using llvm::support::ulittle32_t;

namespace lld {
namespace coff {

SectionChunk::SectionChunk(ObjectFile *F, const coff_section *H)
    : Chunk(SectionKind), Repl(this), File(F), Header(H),
      Relocs(File->getCOFFObj()->getRelocations(Header)),
      NumRelocs(std::distance(Relocs.begin(), Relocs.end())) {
  // Initialize SectionName.
  File->getCOFFObj()->getSectionName(Header, SectionName);

  Align = Header->getAlignment();

  // Only COMDAT sections are subject of dead-stripping.
  Live = !isCOMDAT();
}

static void add16(uint8_t *P, int16_t V) { write16le(P, read16le(P) + V); }
static void add32(uint8_t *P, int32_t V) { write32le(P, read32le(P) + V); }
static void add64(uint8_t *P, int64_t V) { write64le(P, read64le(P) + V); }
static void or16(uint8_t *P, uint16_t V) { write16le(P, read16le(P) | V); }

void SectionChunk::applyRelX64(uint8_t *Off, uint16_t Type, Defined *Sym,
                               uint64_t P) const {
  uint64_t S = Sym->getRVA();
  switch (Type) {
  case IMAGE_REL_AMD64_ADDR32:   add32(Off, S + Config->ImageBase); break;
  case IMAGE_REL_AMD64_ADDR64:   add64(Off, S + Config->ImageBase); break;
  case IMAGE_REL_AMD64_ADDR32NB: add32(Off, S); break;
  case IMAGE_REL_AMD64_REL32:    add32(Off, S - P - 4); break;
  case IMAGE_REL_AMD64_REL32_1:  add32(Off, S - P - 5); break;
  case IMAGE_REL_AMD64_REL32_2:  add32(Off, S - P - 6); break;
  case IMAGE_REL_AMD64_REL32_3:  add32(Off, S - P - 7); break;
  case IMAGE_REL_AMD64_REL32_4:  add32(Off, S - P - 8); break;
  case IMAGE_REL_AMD64_REL32_5:  add32(Off, S - P - 9); break;
  case IMAGE_REL_AMD64_SECTION:  add16(Off, Sym->getSectionIndex()); break;
  case IMAGE_REL_AMD64_SECREL:   add32(Off, Sym->getSecrel()); break;
  default:
    fatal("unsupported relocation type");
  }
}

void SectionChunk::applyRelX86(uint8_t *Off, uint16_t Type, Defined *Sym,
                               uint64_t P) const {
  uint64_t S = Sym->getRVA();
  switch (Type) {
  case IMAGE_REL_I386_ABSOLUTE: break;
  case IMAGE_REL_I386_DIR32:    add32(Off, S + Config->ImageBase); break;
  case IMAGE_REL_I386_DIR32NB:  add32(Off, S); break;
  case IMAGE_REL_I386_REL32:    add32(Off, S - P - 4); break;
  case IMAGE_REL_I386_SECTION:  add16(Off, Sym->getSectionIndex()); break;
  case IMAGE_REL_I386_SECREL:   add32(Off, Sym->getSecrel()); break;
  default:
    fatal("unsupported relocation type");
  }
}

static void applyMOV(uint8_t *Off, uint16_t V) {
  or16(Off, ((V & 0x800) >> 1) | ((V >> 12) & 0xf));
  or16(Off + 2, ((V & 0x700) << 4) | (V & 0xff));
}

static void applyMOV32T(uint8_t *Off, uint32_t V) {
  applyMOV(Off, V);           // set MOVW operand
  applyMOV(Off + 4, V >> 16); // set MOVT operand
}

static void applyBranch20T(uint8_t *Off, int32_t V) {
  uint32_t S = V < 0 ? 1 : 0;
  uint32_t J1 = (V >> 19) & 1;
  uint32_t J2 = (V >> 18) & 1;
  or16(Off, (S << 10) | ((V >> 12) & 0x3f));
  or16(Off + 2, (J1 << 13) | (J2 << 11) | ((V >> 1) & 0x7ff));
}

static void applyBranch24T(uint8_t *Off, int32_t V) {
  uint32_t S = V < 0 ? 1 : 0;
  uint32_t J1 = ((~V >> 23) & 1) ^ S;
  uint32_t J2 = ((~V >> 22) & 1) ^ S;
  or16(Off, (S << 10) | ((V >> 12) & 0x3ff));
  or16(Off + 2, (J1 << 13) | (J2 << 11) | ((V >> 1) & 0x7ff));
}

void SectionChunk::applyRelARM(uint8_t *Off, uint16_t Type, Defined *Sym,
                               uint64_t P) const {
  uint64_t S = Sym->getRVA();
  // Pointer to thumb code must have the LSB set.
  if (Sym->isExecutable())
    S |= 1;
  switch (Type) {
  case IMAGE_REL_ARM_ADDR32:    add32(Off, S + Config->ImageBase); break;
  case IMAGE_REL_ARM_ADDR32NB:  add32(Off, S); break;
  case IMAGE_REL_ARM_MOV32T:    applyMOV32T(Off, S + Config->ImageBase); break;
  case IMAGE_REL_ARM_BRANCH20T: applyBranch20T(Off, S - P - 4); break;
  case IMAGE_REL_ARM_BRANCH24T: applyBranch24T(Off, S - P - 4); break;
  case IMAGE_REL_ARM_BLX23T:    applyBranch24T(Off, S - P - 4); break;
  default:
    fatal("unsupported relocation type");
  }
}

void SectionChunk::writeTo(uint8_t *Buf) const {
  if (!hasData())
    return;
  // Copy section contents from source object file to output file.
  ArrayRef<uint8_t> A = getContents();
  memcpy(Buf + OutputSectionOff, A.data(), A.size());

  // Apply relocations.
  for (const coff_relocation &Rel : Relocs) {
    uint8_t *Off = Buf + OutputSectionOff + Rel.VirtualAddress;
    SymbolBody *Body = File->getSymbolBody(Rel.SymbolTableIndex)->repl();
    Defined *Sym = cast<Defined>(Body);
    uint64_t P = RVA + Rel.VirtualAddress;
    switch (Config->Machine) {
    case AMD64:
      applyRelX64(Off, Rel.Type, Sym, P);
      break;
    case I386:
      applyRelX86(Off, Rel.Type, Sym, P);
      break;
    case ARMNT:
      applyRelARM(Off, Rel.Type, Sym, P);
      break;
    default:
      llvm_unreachable("unknown machine type");
    }
  }
}

void SectionChunk::addAssociative(SectionChunk *Child) {
  AssocChildren.push_back(Child);
}

static uint8_t getBaserelType(const coff_relocation &Rel) {
  switch (Config->Machine) {
  case AMD64:
    if (Rel.Type == IMAGE_REL_AMD64_ADDR64)
      return IMAGE_REL_BASED_DIR64;
    return IMAGE_REL_BASED_ABSOLUTE;
  case I386:
    if (Rel.Type == IMAGE_REL_I386_DIR32)
      return IMAGE_REL_BASED_HIGHLOW;
    return IMAGE_REL_BASED_ABSOLUTE;
  case ARMNT:
    if (Rel.Type == IMAGE_REL_ARM_ADDR32)
      return IMAGE_REL_BASED_HIGHLOW;
    if (Rel.Type == IMAGE_REL_ARM_MOV32T)
      return IMAGE_REL_BASED_ARM_MOV32T;
    return IMAGE_REL_BASED_ABSOLUTE;
  default:
    llvm_unreachable("unknown machine type");
  }
}

// Windows-specific.
// Collect all locations that contain absolute addresses, which need to be
// fixed by the loader if load-time relocation is needed.
// Only called when base relocation is enabled.
void SectionChunk::getBaserels(std::vector<Baserel> *Res) {
  for (const coff_relocation &Rel : Relocs) {
    uint8_t Ty = getBaserelType(Rel);
    if (Ty == IMAGE_REL_BASED_ABSOLUTE)
      continue;
    SymbolBody *Body = File->getSymbolBody(Rel.SymbolTableIndex)->repl();
    if (isa<DefinedAbsolute>(Body))
      continue;
    Res->emplace_back(RVA + Rel.VirtualAddress, Ty);
  }
}

bool SectionChunk::hasData() const {
  return !(Header->Characteristics & IMAGE_SCN_CNT_UNINITIALIZED_DATA);
}

uint32_t SectionChunk::getPermissions() const {
  return Header->Characteristics & PermMask;
}

bool SectionChunk::isCOMDAT() const {
  return Header->Characteristics & IMAGE_SCN_LNK_COMDAT;
}

void SectionChunk::printDiscardedMessage() const {
  // Removed by dead-stripping. If it's removed by ICF, ICF already
  // printed out the name, so don't repeat that here.
  if (Sym && this == Repl)
    llvm::outs() << "Discarded " << Sym->getName() << "\n";
}

StringRef SectionChunk::getDebugName() {
  if (Sym)
    return Sym->getName();
  return "";
}

ArrayRef<uint8_t> SectionChunk::getContents() const {
  ArrayRef<uint8_t> A;
  File->getCOFFObj()->getSectionContents(Header, A);
  return A;
}

void SectionChunk::replace(SectionChunk *Other) {
  Other->Repl = Repl;
  Other->Live = false;
}

CommonChunk::CommonChunk(const COFFSymbolRef S) : Sym(S) {
  // Common symbols are aligned on natural boundaries up to 32 bytes.
  // This is what MSVC link.exe does.
  Align = std::min(uint64_t(32), NextPowerOf2(Sym.getValue()));
}

uint32_t CommonChunk::getPermissions() const {
  return IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ |
         IMAGE_SCN_MEM_WRITE;
}

void StringChunk::writeTo(uint8_t *Buf) const {
  memcpy(Buf + OutputSectionOff, Str.data(), Str.size());
}

ImportThunkChunkX64::ImportThunkChunkX64(Defined *S) : ImpSymbol(S) {
  // Intel Optimization Manual says that all branch targets
  // should be 16-byte aligned. MSVC linker does this too.
  Align = 16;
}

void ImportThunkChunkX64::writeTo(uint8_t *Buf) const {
  memcpy(Buf + OutputSectionOff, ImportThunkX86, sizeof(ImportThunkX86));
  // The first two bytes is a JMP instruction. Fill its operand.
  write32le(Buf + OutputSectionOff + 2, ImpSymbol->getRVA() - RVA - getSize());
}

void ImportThunkChunkX86::getBaserels(std::vector<Baserel> *Res) {
  Res->emplace_back(getRVA() + 2);
}

void ImportThunkChunkX86::writeTo(uint8_t *Buf) const {
  memcpy(Buf + OutputSectionOff, ImportThunkX86, sizeof(ImportThunkX86));
  // The first two bytes is a JMP instruction. Fill its operand.
  write32le(Buf + OutputSectionOff + 2,
            ImpSymbol->getRVA() + Config->ImageBase);
}

void ImportThunkChunkARM::getBaserels(std::vector<Baserel> *Res) {
  Res->emplace_back(getRVA(), IMAGE_REL_BASED_ARM_MOV32T);
}

void ImportThunkChunkARM::writeTo(uint8_t *Buf) const {
  memcpy(Buf + OutputSectionOff, ImportThunkARM, sizeof(ImportThunkARM));
  // Fix mov.w and mov.t operands.
  applyMOV32T(Buf + OutputSectionOff, ImpSymbol->getRVA() + Config->ImageBase);
}

void LocalImportChunk::getBaserels(std::vector<Baserel> *Res) {
  Res->emplace_back(getRVA());
}

size_t LocalImportChunk::getSize() const {
  return Config->is64() ? 8 : 4;
}

void LocalImportChunk::writeTo(uint8_t *Buf) const {
  if (Config->is64()) {
    write64le(Buf + OutputSectionOff, Sym->getRVA() + Config->ImageBase);
  } else {
    write32le(Buf + OutputSectionOff, Sym->getRVA() + Config->ImageBase);
  }
}

void SEHTableChunk::writeTo(uint8_t *Buf) const {
  ulittle32_t *Begin = reinterpret_cast<ulittle32_t *>(Buf + OutputSectionOff);
  size_t Cnt = 0;
  for (Defined *D : Syms)
    Begin[Cnt++] = D->getRVA();
  std::sort(Begin, Begin + Cnt);
}

// Windows-specific.
// This class represents a block in .reloc section.
BaserelChunk::BaserelChunk(uint32_t Page, Baserel *Begin, Baserel *End) {
  // Block header consists of 4 byte page RVA and 4 byte block size.
  // Each entry is 2 byte. Last entry may be padding.
  Data.resize(alignTo((End - Begin) * 2 + 8, 4));
  uint8_t *P = Data.data();
  write32le(P, Page);
  write32le(P + 4, Data.size());
  P += 8;
  for (Baserel *I = Begin; I != End; ++I) {
    write16le(P, (I->Type << 12) | (I->RVA - Page));
    P += 2;
  }
}

void BaserelChunk::writeTo(uint8_t *Buf) const {
  memcpy(Buf + OutputSectionOff, Data.data(), Data.size());
}

uint8_t Baserel::getDefaultType() {
  switch (Config->Machine) {
  case AMD64:
    return IMAGE_REL_BASED_DIR64;
  case I386:
    return IMAGE_REL_BASED_HIGHLOW;
  default:
    llvm_unreachable("unknown machine type");
  }
}

} // namespace coff
} // namespace lld
