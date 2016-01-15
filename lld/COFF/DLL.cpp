//===- DLL.cpp ------------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines various types of chunks for the DLL import or export
// descriptor tables. They are inherently Windows-specific.
// You need to read Microsoft PE/COFF spec to understand details
// about the data structures.
//
// If you are not particularly interested in linking against Windows
// DLL, you can skip this file, and you should still be able to
// understand the rest of the linker.
//
//===----------------------------------------------------------------------===//

#include "Chunks.h"
#include "DLL.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::COFF;

namespace lld {
namespace coff {
namespace {

// Import table

static int ptrSize() { return Config->is64() ? 8 : 4; }

// A chunk for the import descriptor table.
class HintNameChunk : public Chunk {
public:
  HintNameChunk(StringRef N, uint16_t H) : Name(N), Hint(H) {}

  size_t getSize() const override {
    // Starts with 2 byte Hint field, followed by a null-terminated string,
    // ends with 0 or 1 byte padding.
    return alignTo(Name.size() + 3, 2);
  }

  void writeTo(uint8_t *Buf) const override {
    write16le(Buf + OutputSectionOff, Hint);
    memcpy(Buf + OutputSectionOff + 2, Name.data(), Name.size());
  }

private:
  StringRef Name;
  uint16_t Hint;
};

// A chunk for the import descriptor table.
class LookupChunk : public Chunk {
public:
  explicit LookupChunk(Chunk *C) : HintName(C) {}
  size_t getSize() const override { return ptrSize(); }

  void writeTo(uint8_t *Buf) const override {
    write32le(Buf + OutputSectionOff, HintName->getRVA());
  }

  Chunk *HintName;
};

// A chunk for the import descriptor table.
// This chunk represent import-by-ordinal symbols.
// See Microsoft PE/COFF spec 7.1. Import Header for details.
class OrdinalOnlyChunk : public Chunk {
public:
  explicit OrdinalOnlyChunk(uint16_t V) : Ordinal(V) {}
  size_t getSize() const override { return ptrSize(); }

  void writeTo(uint8_t *Buf) const override {
    // An import-by-ordinal slot has MSB 1 to indicate that
    // this is import-by-ordinal (and not import-by-name).
    if (Config->is64()) {
      write64le(Buf + OutputSectionOff, (1ULL << 63) | Ordinal);
    } else {
      write32le(Buf + OutputSectionOff, (1ULL << 31) | Ordinal);
    }
  }

  uint16_t Ordinal;
};

// A chunk for the import descriptor table.
class ImportDirectoryChunk : public Chunk {
public:
  explicit ImportDirectoryChunk(Chunk *N) : DLLName(N) {}
  size_t getSize() const override { return sizeof(ImportDirectoryTableEntry); }

  void writeTo(uint8_t *Buf) const override {
    auto *E = (coff_import_directory_table_entry *)(Buf + OutputSectionOff);
    E->ImportLookupTableRVA = LookupTab->getRVA();
    E->NameRVA = DLLName->getRVA();
    E->ImportAddressTableRVA = AddressTab->getRVA();
  }

  Chunk *DLLName;
  Chunk *LookupTab;
  Chunk *AddressTab;
};

// A chunk representing null terminator in the import table.
// Contents of this chunk is always null bytes.
class NullChunk : public Chunk {
public:
  explicit NullChunk(size_t N) : Size(N) {}
  bool hasData() const override { return false; }
  size_t getSize() const override { return Size; }
  void setAlign(size_t N) { Align = N; }

private:
  size_t Size;
};

static std::vector<std::vector<DefinedImportData *>>
binImports(const std::vector<DefinedImportData *> &Imports) {
  // Group DLL-imported symbols by DLL name because that's how
  // symbols are layed out in the import descriptor table.
  auto Less = [](const std::string &A, const std::string &B) {
    return Config->DLLOrder[A] < Config->DLLOrder[B];
  };
  std::map<std::string, std::vector<DefinedImportData *>,
           bool(*)(const std::string &, const std::string &)> M(Less);
  for (DefinedImportData *Sym : Imports)
    M[Sym->getDLLName().lower()].push_back(Sym);

  std::vector<std::vector<DefinedImportData *>> V;
  for (auto &P : M) {
    // Sort symbols by name for each group.
    std::vector<DefinedImportData *> &Syms = P.second;
    std::sort(Syms.begin(), Syms.end(),
              [](DefinedImportData *A, DefinedImportData *B) {
                return A->getName() < B->getName();
              });
    V.push_back(std::move(Syms));
  }
  return V;
}

// Export table
// See Microsoft PE/COFF spec 4.3 for details.

// A chunk for the delay import descriptor table etnry.
class DelayDirectoryChunk : public Chunk {
public:
  explicit DelayDirectoryChunk(Chunk *N) : DLLName(N) {}

  size_t getSize() const override {
    return sizeof(delay_import_directory_table_entry);
  }

  void writeTo(uint8_t *Buf) const override {
    auto *E = (delay_import_directory_table_entry *)(Buf + OutputSectionOff);
    E->Attributes = 1;
    E->Name = DLLName->getRVA();
    E->ModuleHandle = ModuleHandle->getRVA();
    E->DelayImportAddressTable = AddressTab->getRVA();
    E->DelayImportNameTable = NameTab->getRVA();
  }

  Chunk *DLLName;
  Chunk *ModuleHandle;
  Chunk *AddressTab;
  Chunk *NameTab;
};

// Initial contents for delay-loaded functions.
// This code calls __delayLoadHelper2 function to resolve a symbol
// and then overwrites its jump table slot with the result
// for subsequent function calls.
static const uint8_t ThunkX64[] = {
    0x51,                               // push    rcx
    0x52,                               // push    rdx
    0x41, 0x50,                         // push    r8
    0x41, 0x51,                         // push    r9
    0x48, 0x83, 0xEC, 0x48,             // sub     rsp, 48h
    0x66, 0x0F, 0x7F, 0x04, 0x24,       // movdqa  xmmword ptr [rsp], xmm0
    0x66, 0x0F, 0x7F, 0x4C, 0x24, 0x10, // movdqa  xmmword ptr [rsp+10h], xmm1
    0x66, 0x0F, 0x7F, 0x54, 0x24, 0x20, // movdqa  xmmword ptr [rsp+20h], xmm2
    0x66, 0x0F, 0x7F, 0x5C, 0x24, 0x30, // movdqa  xmmword ptr [rsp+30h], xmm3
    0x48, 0x8D, 0x15, 0, 0, 0, 0,       // lea     rdx, [__imp_<FUNCNAME>]
    0x48, 0x8D, 0x0D, 0, 0, 0, 0,       // lea     rcx, [___DELAY_IMPORT_...]
    0xE8, 0, 0, 0, 0,                   // call    __delayLoadHelper2
    0x66, 0x0F, 0x6F, 0x04, 0x24,       // movdqa  xmm0, xmmword ptr [rsp]
    0x66, 0x0F, 0x6F, 0x4C, 0x24, 0x10, // movdqa  xmm1, xmmword ptr [rsp+10h]
    0x66, 0x0F, 0x6F, 0x54, 0x24, 0x20, // movdqa  xmm2, xmmword ptr [rsp+20h]
    0x66, 0x0F, 0x6F, 0x5C, 0x24, 0x30, // movdqa  xmm3, xmmword ptr [rsp+30h]
    0x48, 0x83, 0xC4, 0x48,             // add     rsp, 48h
    0x41, 0x59,                         // pop     r9
    0x41, 0x58,                         // pop     r8
    0x5A,                               // pop     rdx
    0x59,                               // pop     rcx
    0xFF, 0xE0,                         // jmp     rax
};

static const uint8_t ThunkX86[] = {
    0x51,              // push  ecx
    0x52,              // push  edx
    0x68, 0, 0, 0, 0,  // push  offset ___imp__<FUNCNAME>
    0x68, 0, 0, 0, 0,  // push  offset ___DELAY_IMPORT_DESCRIPTOR_<DLLNAME>_dll
    0xE8, 0, 0, 0, 0,  // call  ___delayLoadHelper2@8
    0x5A,              // pop   edx
    0x59,              // pop   ecx
    0xFF, 0xE0,        // jmp   eax
};

// A chunk for the delay import thunk.
class ThunkChunkX64 : public Chunk {
public:
  ThunkChunkX64(Defined *I, Chunk *D, Defined *H)
      : Imp(I), Desc(D), Helper(H) {}

  size_t getSize() const override { return sizeof(ThunkX64); }

  void writeTo(uint8_t *Buf) const override {
    memcpy(Buf + OutputSectionOff, ThunkX64, sizeof(ThunkX64));
    write32le(Buf + OutputSectionOff + 36, Imp->getRVA() - RVA - 40);
    write32le(Buf + OutputSectionOff + 43, Desc->getRVA() - RVA - 47);
    write32le(Buf + OutputSectionOff + 48, Helper->getRVA() - RVA - 52);
  }

  Defined *Imp = nullptr;
  Chunk *Desc = nullptr;
  Defined *Helper = nullptr;
};

class ThunkChunkX86 : public Chunk {
public:
  ThunkChunkX86(Defined *I, Chunk *D, Defined *H)
      : Imp(I), Desc(D), Helper(H) {}

  size_t getSize() const override { return sizeof(ThunkX86); }

  void writeTo(uint8_t *Buf) const override {
    memcpy(Buf + OutputSectionOff, ThunkX86, sizeof(ThunkX86));
    write32le(Buf + OutputSectionOff + 3, Imp->getRVA() + Config->ImageBase);
    write32le(Buf + OutputSectionOff + 8, Desc->getRVA() + Config->ImageBase);
    write32le(Buf + OutputSectionOff + 13, Helper->getRVA() - RVA - 17);
  }

  void getBaserels(std::vector<Baserel> *Res) override {
    Res->emplace_back(RVA + 3);
    Res->emplace_back(RVA + 8);
  }

  Defined *Imp = nullptr;
  Chunk *Desc = nullptr;
  Defined *Helper = nullptr;
};

// A chunk for the import descriptor table.
class DelayAddressChunk : public Chunk {
public:
  explicit DelayAddressChunk(Chunk *C) : Thunk(C) {}
  size_t getSize() const override { return ptrSize(); }

  void writeTo(uint8_t *Buf) const override {
    if (Config->is64()) {
      write64le(Buf + OutputSectionOff, Thunk->getRVA() + Config->ImageBase);
    } else {
      write32le(Buf + OutputSectionOff, Thunk->getRVA() + Config->ImageBase);
    }
  }

  void getBaserels(std::vector<Baserel> *Res) override {
    Res->emplace_back(RVA);
  }

  Chunk *Thunk;
};

// Export table
// Read Microsoft PE/COFF spec 5.3 for details.

// A chunk for the export descriptor table.
class ExportDirectoryChunk : public Chunk {
public:
  ExportDirectoryChunk(int I, int J, Chunk *D, Chunk *A, Chunk *N, Chunk *O)
      : MaxOrdinal(I), NameTabSize(J), DLLName(D), AddressTab(A), NameTab(N),
        OrdinalTab(O) {}

  size_t getSize() const override {
    return sizeof(export_directory_table_entry);
  }

  void writeTo(uint8_t *Buf) const override {
    auto *E = (export_directory_table_entry *)(Buf + OutputSectionOff);
    E->NameRVA = DLLName->getRVA();
    E->OrdinalBase = 0;
    E->AddressTableEntries = MaxOrdinal + 1;
    E->NumberOfNamePointers = NameTabSize;
    E->ExportAddressTableRVA = AddressTab->getRVA();
    E->NamePointerRVA = NameTab->getRVA();
    E->OrdinalTableRVA = OrdinalTab->getRVA();
  }

  uint16_t MaxOrdinal;
  uint16_t NameTabSize;
  Chunk *DLLName;
  Chunk *AddressTab;
  Chunk *NameTab;
  Chunk *OrdinalTab;
};

class AddressTableChunk : public Chunk {
public:
  explicit AddressTableChunk(size_t MaxOrdinal) : Size(MaxOrdinal + 1) {}
  size_t getSize() const override { return Size * 4; }

  void writeTo(uint8_t *Buf) const override {
    for (Export &E : Config->Exports) {
      uint8_t *P = Buf + OutputSectionOff + E.Ordinal * 4;
      if (E.ForwardChunk) {
        write32le(P, E.ForwardChunk->getRVA());
      } else {
        write32le(P, cast<Defined>(E.Sym->repl())->getRVA());
      }
    }
  }

private:
  size_t Size;
};

class NamePointersChunk : public Chunk {
public:
  explicit NamePointersChunk(std::vector<Chunk *> &V) : Chunks(V) {}
  size_t getSize() const override { return Chunks.size() * 4; }

  void writeTo(uint8_t *Buf) const override {
    uint8_t *P = Buf + OutputSectionOff;
    for (Chunk *C : Chunks) {
      write32le(P, C->getRVA());
      P += 4;
    }
  }

private:
  std::vector<Chunk *> Chunks;
};

class ExportOrdinalChunk : public Chunk {
public:
  explicit ExportOrdinalChunk(size_t I) : Size(I) {}
  size_t getSize() const override { return Size * 2; }

  void writeTo(uint8_t *Buf) const override {
    uint8_t *P = Buf + OutputSectionOff;
    for (Export &E : Config->Exports) {
      if (E.Noname)
        continue;
      write16le(P, E.Ordinal);
      P += 2;
    }
  }

private:
  size_t Size;
};

} // anonymous namespace

uint64_t IdataContents::getDirSize() {
  return Dirs.size() * sizeof(ImportDirectoryTableEntry);
}

uint64_t IdataContents::getIATSize() {
  return Addresses.size() * ptrSize();
}

// Returns a list of .idata contents.
// See Microsoft PE/COFF spec 5.4 for details.
std::vector<Chunk *> IdataContents::getChunks() {
  create();
  std::vector<Chunk *> V;
  // The loader assumes a specific order of data.
  // Add each type in the correct order.
  for (std::unique_ptr<Chunk> &C : Dirs)
    V.push_back(C.get());
  for (std::unique_ptr<Chunk> &C : Lookups)
    V.push_back(C.get());
  for (std::unique_ptr<Chunk> &C : Addresses)
    V.push_back(C.get());
  for (std::unique_ptr<Chunk> &C : Hints)
    V.push_back(C.get());
  for (auto &P : DLLNames) {
    std::unique_ptr<Chunk> &C = P.second;
    V.push_back(C.get());
  }
  return V;
}

void IdataContents::create() {
  std::vector<std::vector<DefinedImportData *>> V = binImports(Imports);

  // Create .idata contents for each DLL.
  for (std::vector<DefinedImportData *> &Syms : V) {
    StringRef Name = Syms[0]->getDLLName();

    // Create lookup and address tables. If they have external names,
    // we need to create HintName chunks to store the names.
    // If they don't (if they are import-by-ordinals), we store only
    // ordinal values to the table.
    size_t Base = Lookups.size();
    for (DefinedImportData *S : Syms) {
      uint16_t Ord = S->getOrdinal();
      if (S->getExternalName().empty()) {
        Lookups.push_back(make_unique<OrdinalOnlyChunk>(Ord));
        Addresses.push_back(make_unique<OrdinalOnlyChunk>(Ord));
        continue;
      }
      auto C = make_unique<HintNameChunk>(S->getExternalName(), Ord);
      Lookups.push_back(make_unique<LookupChunk>(C.get()));
      Addresses.push_back(make_unique<LookupChunk>(C.get()));
      Hints.push_back(std::move(C));
    }
    // Terminate with null values.
    Lookups.push_back(make_unique<NullChunk>(ptrSize()));
    Addresses.push_back(make_unique<NullChunk>(ptrSize()));

    for (int I = 0, E = Syms.size(); I < E; ++I)
      Syms[I]->setLocation(Addresses[Base + I].get());

    // Create the import table header.
    if (!DLLNames.count(Name))
      DLLNames[Name] = make_unique<StringChunk>(Name);
    auto Dir = make_unique<ImportDirectoryChunk>(DLLNames[Name].get());
    Dir->LookupTab = Lookups[Base].get();
    Dir->AddressTab = Addresses[Base].get();
    Dirs.push_back(std::move(Dir));
  }
  // Add null terminator.
  Dirs.push_back(make_unique<NullChunk>(sizeof(ImportDirectoryTableEntry)));
}

std::vector<Chunk *> DelayLoadContents::getChunks() {
  std::vector<Chunk *> V;
  for (std::unique_ptr<Chunk> &C : Dirs)
    V.push_back(C.get());
  for (std::unique_ptr<Chunk> &C : Names)
    V.push_back(C.get());
  for (std::unique_ptr<Chunk> &C : HintNames)
    V.push_back(C.get());
  for (auto &P : DLLNames) {
    std::unique_ptr<Chunk> &C = P.second;
    V.push_back(C.get());
  }
  return V;
}

std::vector<Chunk *> DelayLoadContents::getDataChunks() {
  std::vector<Chunk *> V;
  for (std::unique_ptr<Chunk> &C : ModuleHandles)
    V.push_back(C.get());
  for (std::unique_ptr<Chunk> &C : Addresses)
    V.push_back(C.get());
  return V;
}

uint64_t DelayLoadContents::getDirSize() {
  return Dirs.size() * sizeof(delay_import_directory_table_entry);
}

void DelayLoadContents::create(Defined *H) {
  Helper = H;
  std::vector<std::vector<DefinedImportData *>> V = binImports(Imports);

  // Create .didat contents for each DLL.
  for (std::vector<DefinedImportData *> &Syms : V) {
    StringRef Name = Syms[0]->getDLLName();

    // Create the delay import table header.
    if (!DLLNames.count(Name))
      DLLNames[Name] = make_unique<StringChunk>(Name);
    auto Dir = make_unique<DelayDirectoryChunk>(DLLNames[Name].get());

    size_t Base = Addresses.size();
    for (DefinedImportData *S : Syms) {
      Chunk *T = newThunkChunk(S, Dir.get());
      auto A = make_unique<DelayAddressChunk>(T);
      Addresses.push_back(std::move(A));
      Thunks.push_back(std::unique_ptr<Chunk>(T));
      StringRef ExtName = S->getExternalName();
      if (ExtName.empty()) {
        Names.push_back(make_unique<OrdinalOnlyChunk>(S->getOrdinal()));
      } else {
        auto C = make_unique<HintNameChunk>(ExtName, 0);
        Names.push_back(make_unique<LookupChunk>(C.get()));
        HintNames.push_back(std::move(C));
      }
    }
    // Terminate with null values.
    Addresses.push_back(make_unique<NullChunk>(8));
    Names.push_back(make_unique<NullChunk>(8));

    for (int I = 0, E = Syms.size(); I < E; ++I)
      Syms[I]->setLocation(Addresses[Base + I].get());
    auto *MH = new NullChunk(8);
    MH->setAlign(8);
    ModuleHandles.push_back(std::unique_ptr<Chunk>(MH));

    // Fill the delay import table header fields.
    Dir->ModuleHandle = MH;
    Dir->AddressTab = Addresses[Base].get();
    Dir->NameTab = Names[Base].get();
    Dirs.push_back(std::move(Dir));
  }
  // Add null terminator.
  Dirs.push_back(
      make_unique<NullChunk>(sizeof(delay_import_directory_table_entry)));
}

Chunk *DelayLoadContents::newThunkChunk(DefinedImportData *S, Chunk *Dir) {
  switch (Config->Machine) {
  case AMD64:
    return new ThunkChunkX64(S, Dir, Helper);
  case I386:
    return new ThunkChunkX86(S, Dir, Helper);
  default:
    llvm_unreachable("unsupported machine type");
  }
}

EdataContents::EdataContents() {
  uint16_t MaxOrdinal = 0;
  for (Export &E : Config->Exports)
    MaxOrdinal = std::max(MaxOrdinal, E.Ordinal);

  auto *DLLName = new StringChunk(sys::path::filename(Config->OutputFile));
  auto *AddressTab = new AddressTableChunk(MaxOrdinal);
  std::vector<Chunk *> Names;
  for (Export &E : Config->Exports)
    if (!E.Noname)
      Names.push_back(new StringChunk(E.ExportName));

  std::vector<Chunk *> Forwards;
  for (Export &E : Config->Exports) {
    if (E.ForwardTo.empty())
      continue;
    E.ForwardChunk = new StringChunk(E.ForwardTo);
    Forwards.push_back(E.ForwardChunk);
  }

  auto *NameTab = new NamePointersChunk(Names);
  auto *OrdinalTab = new ExportOrdinalChunk(Names.size());
  auto *Dir = new ExportDirectoryChunk(MaxOrdinal, Names.size(), DLLName,
                                       AddressTab, NameTab, OrdinalTab);
  Chunks.push_back(std::unique_ptr<Chunk>(Dir));
  Chunks.push_back(std::unique_ptr<Chunk>(DLLName));
  Chunks.push_back(std::unique_ptr<Chunk>(AddressTab));
  Chunks.push_back(std::unique_ptr<Chunk>(NameTab));
  Chunks.push_back(std::unique_ptr<Chunk>(OrdinalTab));
  for (Chunk *C : Names)
    Chunks.push_back(std::unique_ptr<Chunk>(C));
  for (Chunk *C : Forwards)
    Chunks.push_back(std::unique_ptr<Chunk>(C));
}

} // namespace coff
} // namespace lld
