//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tracking associated with executable images. Support for global values comes
/// from loading the executable within the virtual address region and marking
/// chunks that overlap with globals in the executable.
///
//===----------------------------------------------------------------------===//

#include "dasan_image.h"

#include <elf.h>

#include "dasan_allocator.h"
#include "dasan_flags.h"
#include "dasan_mapping.h"
#include "dasan_platform.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_offload.h"

using namespace __sanitizer;

namespace __dasan {

namespace {
const char kGuardName[] = "__dasan_guard";

// An executable image that was loaded into virtual memory.
struct TrackedImage {
  uptr Addr;
  uptr LoadSize;
  uptr MapSize;
  u64 Handle;
  bool Recorded;
  bool Described;
};

InternalMmapVectorNoCtor<GlobalInfo> Globals;
InternalMmapVectorNoCtor<TrackedImage> Images;

bool InFile(uptr Size, const Elf64_Shdr& S) {
  return S.sh_type != SHT_NOBITS && S.sh_offset <= Size &&
         S.sh_size <= Size - S.sh_offset;
}

// Only objects with valid symbols are instrumented by the runtime.
bool DefinedObject(const Elf64_Ehdr& Hdr, const Elf64_Shdr* Sections,
                   const Elf64_Sym& Sym, uptr LoadSize, uptr Chunk) {
  if (ELF64_ST_TYPE(Sym.st_info) != STT_OBJECT || !Sym.st_size)
    return false;
  if (Sym.st_shndx == SHN_UNDEF || Sym.st_shndx >= Hdr.e_shnum)
    return false;
  if (!(Sections[Sym.st_shndx].sh_flags & SHF_ALLOC))
    return false;
  if (Sym.st_value % Chunk)
    return false;
  if (Sym.st_value > LoadSize || Sym.st_size > LoadSize - Sym.st_value)
    return false;
  return true;
}

// The pass does not pad __* / .* symbols or kernel descriptors, ignore them.
bool HasRedzone(const char* Name) {
  if (!Name)
    return true;
  if (Name[0] == '.' || (Name[0] == '_' && Name[1] == '_'))
    return false;
  uptr N = internal_strlen(Name);
  return N < 3 || internal_strncmp(Name + N - 3, ".kd", 3) != 0;
}

// Iterate the executable's symbol table to get each symbol's size and offset.
// The HSA symbol interface only exposes the dynamic symbol table.
void DescribeGlobals(uptr Base, uptr LoadSize, const void* Storage,
                     uptr StorageSize, uptr& Described) {
  if (!Storage || StorageSize < sizeof(Elf64_Ehdr) || !LoadSize)
    return;
  auto* Hdr = reinterpret_cast<const Elf64_Ehdr*>(Storage);
  if (internal_memcmp(Hdr->e_ident, ELFMAG, SELFMAG) != 0 ||
      Hdr->e_ident[EI_CLASS] != ELFCLASS64 ||
      Hdr->e_shentsize != sizeof(Elf64_Shdr) || !Hdr->e_shnum)
    return;
  const uptr ShdrBytes = Hdr->e_shnum * sizeof(Elf64_Shdr);
  if (Hdr->e_shoff > StorageSize || ShdrBytes > StorageSize - Hdr->e_shoff)
    return;
  auto* Sections = reinterpret_cast<const Elf64_Shdr*>(
      static_cast<const char*>(Storage) + Hdr->e_shoff);

  // Find the symbol table.
  const Elf64_Shdr* Symtab = nullptr;
  for (uptr S = 0; S < Hdr->e_shnum; ++S) {
    const Elf64_Shdr& Sec = Sections[S];
    if (!InFile(StorageSize, Sec) || Sec.sh_entsize != sizeof(Elf64_Sym))
      continue;
    if (Sec.sh_type == SHT_SYMTAB)
      Symtab = &Sec;
    else if (Sec.sh_type == SHT_DYNSYM && !Symtab)
      Symtab = &Sec;
  }
  if (!Symtab || !Symtab->sh_size || Symtab->sh_link >= Hdr->e_shnum)
    return;

  // Get the symbol table and string table offsets in the ELF.
  const Elf64_Shdr& Strtab = Sections[Symtab->sh_link];
  const char* Names = InFile(StorageSize, Strtab)
                          ? static_cast<const char*>(Storage) + Strtab.sh_offset
                          : nullptr;
  auto* Syms = reinterpret_cast<const Elf64_Sym*>(
      static_cast<const char*>(Storage) + Symtab->sh_offset);

  auto NameOf = [&](const Elf64_Sym& Sym) -> const char* {
    return Names && Sym.st_name < Strtab.sh_size ? Names + Sym.st_name
                                                 : nullptr;
  };
  auto IsGuard = [](const char* Name) {
    return Name && !internal_strncmp(Name, kGuardName, sizeof(kGuardName) - 1);
  };

  Allocator& A = GetAllocator();
  const uptr Chunk = ClassIdToSize(0);
  const uptr Count = Symtab->sh_size / sizeof(Elf64_Sym);
  for (uptr N = 0; N < Count; ++N) {
    const Elf64_Sym& Sym = Syms[N];
    if (!DefinedObject(*Hdr, Sections, Sym, LoadSize, Chunk))
      continue;
    const char* Name = NameOf(Sym);
    if (IsGuard(Name))
      continue;
    const uptr Obj = Base + (uptr)Sym.st_value;
    const uptr Size = (uptr)Sym.st_size;
    if (!A.Publish(Obj, Size, HasRedzone(Name)))
      continue;
    Globals.push_back({Obj, Size, Name ? internal_strdup(Name) : nullptr});
    ++Described;
  }
}

void ForgetGlobals() {
  for (uptr N = 0; N < Globals.size(); ++N)
    if (Globals[N].Name)
      InternalFree(const_cast<char*>(Globals[N].Name));
  Globals.clear();
}

void ForgetGlobals(uptr Beg, uptr Size) {
  uptr Dst = 0;
  for (uptr N = 0; N < Globals.size(); ++N) {
    if (Globals[N].Beg >= Beg && Globals[N].Beg - Beg < Size) {
      if (Globals[N].Name)
        InternalFree(const_cast<char*>(Globals[N].Name));
      continue;
    }
    if (Dst != N)
      Globals[Dst] = Globals[N];
    ++Dst;
  }
  Globals.resize(Dst);
}

void DropImage(uptr I) {
  TrackedImage& Img = Images[I];
  VaUnmap(Img.Addr, Img.MapSize);
  VaDestroy(Img.Handle);
  GetAllocator().Release(Img.Addr);
  ForgetGlobals(Img.Addr, Img.LoadSize);
  if (I + 1 != Images.size())
    Images[I] = Images.back();
  Images.pop_back();
}

void DescribePlaced(TrackedImage& Img, const Offload::CodeObject& Obj) {
  if (Img.Described)
    return;
  uptr Described = 0;
  DescribeGlobals(Img.Addr, Obj.LoadSize, Obj.Storage, Obj.StorageSize,
                  Described);
  Img.Described = true;
  VReport(1, "%s: placed 0x%zx (+%zu); %zu globals\n", SanitizerToolName,
          Img.Addr, Img.LoadSize, Described);
}

// The image is already in the window. Publish each global from its symbol.
void OnCodeObject(const Offload::CodeObject& Obj) {
  for (TrackedImage& Img : Images) {
    if (!Img.Recorded || Img.Addr != Obj.LoadBase)
      continue;
    DescribePlaced(Img, Obj);
    return;
  }
}

}  // namespace

bool ReserveExecutable(hsa_agent_t Agent, uptr LoadSize, uptr& Addr,
                       uptr& Span) {
  hsa_amd_memory_pool_t Pool = PoolForAgent(Agent);
  if (!HasPool(Pool) || !LoadSize)
    return false;
  const uptr MapSize = RoundUpTo(LoadSize, VaGranule());
  Allocator& A = GetAllocator();

  // The instrumentation pass pads globals to the smallest size class.
  uptr Base = A.Allocate(Pool, /*ClassId=*/0, MapSize, 0, 0, 0,
                         /*MapUser=*/false);
  if (!Base) {
    ++A.Count.Exhausted;
    return false;
  }
  u64 H = 0;
  if (!VaCreate(Pool, MapSize, HSA_AMD_VMEM_HANDLE_CREATE_EXECUTABLE, &H) ||
      !VaMap(Base, MapSize, H, Pool, 0, /*Executable=*/true)) {
    if (H)
      VaDestroy(H);
    A.Release(Base);
    return false;
  }
  Images.push_back({Base, LoadSize, MapSize, H, false, false});
  Addr = Base;
  Span = MapSize;
  return true;
}

void ReleaseExecutable(uptr Addr) {
  for (uptr I = 0; I < Images.size(); ++I) {
    if (Images[I].Addr != Addr || Images[I].Recorded)
      continue;
    DropImage(I);
    return;
  }
}

void RecordExecutable(uptr Addr) {
  for (TrackedImage& Img : Images) {
    if (Img.Addr != Addr)
      continue;
    Img.Recorded = true;
    return;
  }
}

void TrackExecutable(hsa_executable_t Exec) {
  if (!flags()->alias_globals)
    return;
  Offload::Get().ForEachCodeObject(Exec, OnCodeObject);
}

void UntrackExecutable(hsa_executable_t Exec) {
  Offload::Get().ForEachCodeObject(Exec, [](const Offload::CodeObject& Obj) {
    for (uptr I = 0; I < Images.size(); ++I) {
      if (!Images[I].Recorded || Images[I].Addr != Obj.LoadBase)
        continue;
      VReport(1, "%s: released 0x%zx (+%zu)\n", SanitizerToolName,
              Images[I].Addr, Images[I].LoadSize);
      DropImage(I);
      return;
    }
  });
}

void UntrackAll() {
  while (Images.size()) DropImage(Images.size() - 1);
  ForgetGlobals();
}

bool LookupGlobal(uptr Beg, GlobalInfo& Out) {
  for (const GlobalInfo& G : Globals) {
    if (G.Beg == Beg) {
      Out = G;
      return true;
    }
  }
  return false;
}

}  // namespace __dasan
