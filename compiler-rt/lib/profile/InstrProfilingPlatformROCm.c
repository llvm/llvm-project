//===- InstrProfilingPlatformROCm.c - Profile data ROCm platform ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include "InstrProfilingPort.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- */
/*  Platform abstraction for dynamic library loading                          */
/* -------------------------------------------------------------------------- */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef HMODULE DylibHandle;

static DylibHandle DylibOpen(const char *Name) {
  return LoadLibraryA(Name);
}

static void *DylibSym(DylibHandle H, const char *Sym) {
  return (void *)(uintptr_t)GetProcAddress(H, Sym);
}

static int DylibAvailable(void) { return 1; }

#else /* POSIX */
#include <dlfcn.h>

/* Use weak references for dl* functions to avoid requiring -ldl at link time.
 *
 * The profile runtime is a static library, so its dependencies must be
 * explicitly linked by the user. Unlike sanitizer runtimes (which are often
 * shared libraries with their own dependencies), adding -ldl globally would
 * affect all profiling users, including those not using HIP/ROCm.
 *
 * With weak references:
 * - Programs without -ldl link successfully (dl* resolve to NULL)
 * - HIP programs get -ldl from the HIP runtime, so dl* work normally
 * - OpenMP offload programs without HIP gracefully skip device profiling
 */
#pragma weak dlopen
#pragma weak dlsym
#pragma weak dlerror

typedef void *DylibHandle;

static DylibHandle DylibOpen(const char *Name) {
  return dlopen(Name, RTLD_LAZY | RTLD_LOCAL);
}

static void *DylibSym(DylibHandle H, const char *Sym) {
  return dlsym(H, Sym);
}

static int DylibAvailable(void) { return dlopen != NULL; }

#endif /* _WIN32 */

static int ProcessDeviceOffloadPrf(void *DeviceOffloadPrf, int TUIndex);

static int IsVerboseMode() {
  static int IsVerbose = -1;
  if (IsVerbose == -1)
    IsVerbose = getenv("LLVM_PROFILE_VERBOSE") != NULL;
  return IsVerbose;
}

/* -------------------------------------------------------------------------- */
/*  Dynamic loading of HIP runtime symbols                                   */
/* -------------------------------------------------------------------------- */

typedef int (*hipGetSymbolAddressTy)(void **, const void *);
typedef int (*hipMemcpyTy)(void *, void *, size_t, int);
typedef int (*hipModuleGetGlobalTy)(void **, size_t *, void *, const char *);

static hipGetSymbolAddressTy pHipGetSymbolAddress = NULL;
static hipMemcpyTy pHipMemcpy = NULL;
static hipModuleGetGlobalTy pHipModuleGetGlobal = NULL;

/* -------------------------------------------------------------------------- */
/*  Device-to-host copies                                                     */
/*  Keep HIP-only to avoid an HSA dependency.                                 */
/* -------------------------------------------------------------------------- */

static void EnsureHipLoaded(void) {
  static int Initialized = 0;
  if (Initialized)
    return;
  Initialized = 1;

  if (!DylibAvailable()) {
    if (IsVerboseMode())
      PROF_NOTE("%s",
                "Dynamic library loading not available - "
                "HIP profiling disabled\n");
    return;
  }

#ifdef _WIN32
  static const char HipLibName[] = "amdhip64.dll";
#else
  static const char HipLibName[] = "libamdhip64.so";
#endif

  DylibHandle Handle = DylibOpen(HipLibName);
  if (!Handle) {
#ifndef _WIN32
    if (dlerror)
      fprintf(stderr, "compiler-rt: failed to open %s: %s\n", HipLibName,
              dlerror());
#endif
    return;
  }

  pHipGetSymbolAddress =
      (hipGetSymbolAddressTy)DylibSym(Handle, "hipGetSymbolAddress");
  pHipMemcpy = (hipMemcpyTy)DylibSym(Handle, "hipMemcpy");
  pHipModuleGetGlobal =
      (hipModuleGetGlobalTy)DylibSym(Handle, "hipModuleGetGlobal");
}

/* -------------------------------------------------------------------------- */
/*  Public wrappers that forward to the loaded HIP symbols                   */
/* -------------------------------------------------------------------------- */

static int hipGetSymbolAddress(void **devPtr, const void *symbol) {
  EnsureHipLoaded();
  return pHipGetSymbolAddress ? pHipGetSymbolAddress(devPtr, symbol) : -1;
}

static int hipMemcpy(void *dest, void *src, size_t len, int kind /*2=DToH*/) {
  EnsureHipLoaded();
  return pHipMemcpy ? pHipMemcpy(dest, src, len, kind) : -1;
}

/* Copy from device to host using HIP.
 * This requires that the device section symbols are registered with CLR,
 * otherwise hipMemcpy may attempt a CPU path and crash. */
static int memcpyDeviceToHost(void *Dst, void *Src, size_t Size) {
  return hipMemcpy(Dst, Src, Size, 2 /* DToH */);
}

static int hipModuleGetGlobal(void **DevPtr, size_t *Bytes, void *Module,
                              const char *Name) {
  EnsureHipLoaded();
  return pHipModuleGetGlobal ? pHipModuleGetGlobal(DevPtr, Bytes, Module, Name)
                             : -1;
}

/* -------------------------------------------------------------------------- */
/*  Dynamic module tracking                                                   */
/* -------------------------------------------------------------------------- */

/* Per-TU profile entry inside a dynamic module.
 * A single dynamic module may contain multiple TUs (e.g. -fgpu-rdc). */
typedef struct {
  void *DeviceVar; /* device address of __llvm_offload_prf_<CUID>      */
  int Processed;   /* 0 = not yet collected, 1 = data already copied   */
} OffloadDynamicTUInfo;

/* One entry per hipModuleLoad call. */
typedef struct {
  void *ModulePtr;           /* hipModule_t handle                        */
  OffloadDynamicTUInfo *TUs; /* array of per-TU entries                 */
  int NumTUs;
  int CapTUs;
} OffloadDynamicModuleInfo;

static OffloadDynamicModuleInfo *DynamicModules = NULL;
static int NumDynamicModules = 0;
static int CapDynamicModules = 0;

/* -------------------------------------------------------------------------- */
/*  ELF symbol enumeration (Linux only)                                       */
/*                                                                            */
/*  AMDGPU code objects are always ELF, but <elf.h> is a Linux system header. */
/*  Dynamic module profiling (hipModuleLoadData) is currently Linux-only.      */
/* -------------------------------------------------------------------------- */

#if defined(__linux__)
#include <elf.h>

/* Callback invoked for every matching symbol name found in the ELF image.
 * Return 0 to continue iteration, non-zero to stop. */
typedef int (*SymbolCallback)(const char *Name, void *UserData);

/* If Image is a clang offload bundle (__CLANG_OFFLOAD_BUNDLE__), find the
 * first embedded code object that is a valid ELF and return a pointer to it.
 * Otherwise return Image unchanged. Returns NULL if no ELF is found. */
static const void *UnwrapOffloadBundle(const void *Image) {
  static const char BundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
  if (memcmp(Image, BundleMagic, 24) != 0)
    return Image; /* Not a bundle, return as-is. */

  const char *Buf = (const char *)Image;
  uint64_t NumEntries;
  memcpy(&NumEntries, Buf + 24, sizeof(uint64_t));

  /* Walk the entry table (starts at offset 32). */
  const char *Cursor = Buf + 32;
  for (uint64_t I = 0; I < NumEntries; ++I) {
    uint64_t EntryOffset, EntrySize, IDSize;
    memcpy(&EntryOffset, Cursor, 8); Cursor += 8;
    memcpy(&EntrySize, Cursor, 8);   Cursor += 8;
    memcpy(&IDSize, Cursor, 8);      Cursor += 8;
    /* Skip the entry ID string. */
    Cursor += IDSize;

    /* Check if this entry contains an ELF. */
    if (EntrySize >= sizeof(Elf64_Ehdr)) {
      const Elf64_Ehdr *E = (const Elf64_Ehdr *)(Buf + EntryOffset);
      if (E->e_ident[EI_MAG0] == ELFMAG0 && E->e_ident[EI_MAG1] == ELFMAG1 &&
          E->e_ident[EI_MAG2] == ELFMAG2 && E->e_ident[EI_MAG3] == ELFMAG3) {
        if (IsVerboseMode())
          PROF_NOTE("Unwrapped offload bundle: entry %lu at offset %lu "
                    "(size %lu)\n",
                    (unsigned long)I, (unsigned long)EntryOffset,
                    (unsigned long)EntrySize);
        return (const void *)(Buf + EntryOffset);
      }
    }
  }

  PROF_WARN("%s", "Offload bundle contains no valid ELF entries\n");
  return NULL;
}

/* Parse an AMDGPU code-object ELF and invoke CB for every global symbol whose
 * name starts with PREFIX.  Image may be NULL (e.g. hipModuleLoad from file)
 * or a clang offload bundle containing an ELF;
 * in that case the function unwraps the bundle first. */
static void EnumerateElfSymbols(const void *Image, const char *Prefix,
                                SymbolCallback CB, void *UserData) {
  if (!Image)
    return;

  /* Handle clang offload bundle wrapping. */
  Image = UnwrapOffloadBundle(Image);
  if (!Image)
    return;

  const Elf64_Ehdr *Ehdr = (const Elf64_Ehdr *)Image;
  if (Ehdr->e_ident[EI_MAG0] != ELFMAG0 || Ehdr->e_ident[EI_MAG1] != ELFMAG1 ||
      Ehdr->e_ident[EI_MAG2] != ELFMAG2 || Ehdr->e_ident[EI_MAG3] != ELFMAG3) {
    if (IsVerboseMode())
      PROF_NOTE("%s", "Image is not a valid ELF, skipping enumeration\n");
    return;
  }

  size_t PrefixLen = strlen(Prefix);
  const char *Base = (const char *)Image;
  const Elf64_Shdr *Shdrs = (const Elf64_Shdr *)(Base + Ehdr->e_shoff);

  for (int i = 0; i < Ehdr->e_shnum; ++i) {
    if (Shdrs[i].sh_type != SHT_SYMTAB)
      continue;

    const Elf64_Sym *Syms = (const Elf64_Sym *)(Base + Shdrs[i].sh_offset);
    int NumSyms = Shdrs[i].sh_size / sizeof(Elf64_Sym);
    /* String table is the section referenced by sh_link. */
    const char *StrTab = Base + Shdrs[Shdrs[i].sh_link].sh_offset;

    for (int j = 0; j < NumSyms; ++j) {
      if (Syms[j].st_name == 0)
        continue;
      const char *Name = StrTab + Syms[j].st_name;
      if (strncmp(Name, Prefix, PrefixLen) == 0) {
        if (CB(Name, UserData))
          return;
      }
    }
  }
}

/* State passed through the enumeration callback. */
typedef struct {
  void *Module; /* hipModule_t */
  OffloadDynamicModuleInfo *ModInfo;
} EnumState;

/* Grow the TU array inside a module entry and register one __llvm_offload_prf_*
 * symbol. Also pre-registers the corresponding per-TU section symbols with CLR
 * (needed so hipMemcpy can copy from those device addresses later). */
static int RegisterPrfSymbol(const char *Name, void *UserData) {
  EnumState *S = (EnumState *)UserData;
  OffloadDynamicModuleInfo *MI = S->ModInfo;

  /* Look up the profile structure symbol. */
  void *DeviceVar = NULL;
  size_t Bytes = 0;
  if (hipModuleGetGlobal(&DeviceVar, &Bytes, S->Module, Name) != 0) {
    PROF_WARN("Failed to get symbol %s for module %p\n", Name, S->Module);
    return 0; /* continue */
  }

  if (IsVerboseMode())
    PROF_NOTE("Module %p: found %s -> %p (%zu bytes)\n", S->Module, Name,
              DeviceVar, Bytes);

  /* Grow TU array if needed. */
  if (MI->NumTUs >= MI->CapTUs) {
    int NewCap = MI->CapTUs ? MI->CapTUs * 2 : 4;
    OffloadDynamicTUInfo *New = (OffloadDynamicTUInfo *)realloc(
        MI->TUs, NewCap * sizeof(OffloadDynamicTUInfo));
    if (!New) {
      PROF_ERR("%s\n", "Failed to grow TU array");
      return 0;
    }
    MI->TUs = New;
    MI->CapTUs = NewCap;
  }
  OffloadDynamicTUInfo *TU = &MI->TUs[MI->NumTUs++];
  TU->DeviceVar = DeviceVar;
  TU->Processed = 0;

  /* Derive the CUID suffix from the symbol name.  The name has the form
   * "__llvm_offload_prf_<CUID>", so the suffix (including underscore) starts
   * at offset strlen("__llvm_offload_prf"). */
  const char *Suffix = Name + strlen("__llvm_offload_prf");

  /* Pre-register per-TU section symbols with CLR memory tracking.
   * The section symbol names use the same CUID suffix:
   *   __llvm_prf_c_<CUID>, __llvm_prf_d_<CUID>,
   *   __profu_all_<CUID>, __llvm_prf_nm_<CUID>  */
  static const char *SectionPrefixes[] = {"__llvm_prf_c", "__llvm_prf_d",
                                          "__profu_all", "__llvm_prf_nm"};
  for (int s = 0; s < 4; ++s) {
    char SectionName[256];
    snprintf(SectionName, sizeof(SectionName), "%s%s", SectionPrefixes[s],
             Suffix);
    void *Dummy = NULL;
    size_t DummyBytes = 0;
    int rc = hipModuleGetGlobal(&Dummy, &DummyBytes, S->Module, SectionName);
    if (IsVerboseMode())
      PROF_NOTE("Module %p: lookup %s -> %s (%p, %zu bytes)\n", S->Module,
                SectionName, rc == 0 ? "found" : "not found", Dummy,
                DummyBytes);
  }

  return 0; /* continue enumeration */
}

#endif /* defined(__linux__) */

/* -------------------------------------------------------------------------- */
/*  Registration / un-registration helpers                                   */
/* -------------------------------------------------------------------------- */

void __llvm_profile_offload_register_dynamic_module(int ModuleLoadRc,
                                                    void **Ptr,
                                                    const void *Image) {
  if (IsVerboseMode())
    PROF_NOTE("Registering loaded module %d: rc=%d, module=%p, image=%p\n",
              NumDynamicModules, ModuleLoadRc, *Ptr, Image);

  if (ModuleLoadRc)
    return;

  if (NumDynamicModules >= CapDynamicModules) {
    int NewCap = CapDynamicModules ? CapDynamicModules * 2 : 64;
    OffloadDynamicModuleInfo *New = (OffloadDynamicModuleInfo *)realloc(
        DynamicModules, NewCap * sizeof(OffloadDynamicModuleInfo));
    if (!New) {
      PROF_ERR("%s\n", "Failed to grow dynamic modules array");
      return;
    }
    DynamicModules = New;
    CapDynamicModules = NewCap;
  }

  OffloadDynamicModuleInfo *MI = &DynamicModules[NumDynamicModules++];
  MI->ModulePtr = *Ptr;
  MI->TUs = NULL;
  MI->NumTUs = 0;
  MI->CapTUs = 0;

  /* Enumerate all __llvm_offload_prf_<CUID> symbols in the ELF image.
   * For each one, look it up via hipModuleGetGlobal (which also registers
   * the device address with CLR for later hipMemcpy) and store the entry.
   *
   * ELF parsing requires <elf.h> which is Linux-only.  On other platforms,
   * dynamic module profiling is not yet supported. */
#if defined(__linux__)
  EnumState State = {*Ptr, MI};
  EnumerateElfSymbols(Image, "__llvm_offload_prf_", RegisterPrfSymbol, &State);
#else
  (void)Image;
  if (IsVerboseMode())
    PROF_NOTE("%s",
              "Dynamic module profiling not supported on this platform\n");
#endif

  if (MI->NumTUs == 0) {
    PROF_WARN("No __llvm_offload_prf_* symbols found in module %p\n", *Ptr);
  } else if (IsVerboseMode()) {
    PROF_NOTE("Module %p: registered %d TU(s)\n", *Ptr, MI->NumTUs);
  }
}

void __llvm_profile_offload_unregister_dynamic_module(void *Ptr) {
  for (int i = 0; i < NumDynamicModules; ++i) {
    OffloadDynamicModuleInfo *MI = &DynamicModules[i];

    if (MI->ModulePtr != Ptr)
      continue;

    if (IsVerboseMode())
      PROF_NOTE("Unregistering module %p (%d TUs)\n", MI->ModulePtr,
                MI->NumTUs);

    /* Process every TU in this module. */
    for (int t = 0; t < MI->NumTUs; ++t) {
      OffloadDynamicTUInfo *TU = &MI->TUs[t];
      if (TU->Processed) {
        if (IsVerboseMode())
          PROF_NOTE("Module %p TU %d already processed, skipping\n", Ptr, t);
        continue;
      }
      /* Use a globally unique index as TU index for the output filename. */
      int TUIndex = i * 1000 + t;
      if (TU->DeviceVar) {
        if (ProcessDeviceOffloadPrf(TU->DeviceVar, TUIndex) == 0)
          TU->Processed = 1;
        else
          PROF_WARN("Failed to process profile data for module %p TU %d\n", Ptr,
                    t);
      }
    }
    return;
  }

  if (IsVerboseMode())
    PROF_WARN("Unregister called for unknown module %p\n", Ptr);
}

static void **OffloadShadowVariables = NULL;
static int NumShadowVariables = 0;
static int CapShadowVariables = 0;

void __llvm_profile_offload_register_shadow_variable(void *ptr) {
  if (NumShadowVariables >= CapShadowVariables) {
    int NewCap = CapShadowVariables ? CapShadowVariables * 2 : 64;
    void **New = (void **)realloc(OffloadShadowVariables, NewCap * sizeof(void *));
    if (!New) {
      PROF_ERR("%s\n", "Failed to grow shadow variables array");
      return;
    }
    OffloadShadowVariables = New;
    CapShadowVariables = NewCap;
  }
  if (IsVerboseMode())
    PROF_NOTE("Registering shadow variable %d: %p\n", NumShadowVariables, ptr);
  OffloadShadowVariables[NumShadowVariables++] = ptr;
}

static void **OffloadSectionShadowVariables = NULL;
static int NumSectionShadowVariables = 0;
static int CapSectionShadowVariables = 0;

void __llvm_profile_offload_register_section_shadow_variable(void *ptr) {
  if (NumSectionShadowVariables >= CapSectionShadowVariables) {
    int NewCap = CapSectionShadowVariables ? CapSectionShadowVariables * 2 : 64;
    void **New =
        (void **)realloc(OffloadSectionShadowVariables, NewCap * sizeof(void *));
    if (!New) {
      PROF_ERR("%s\n", "Failed to grow section shadow variables array");
      return;
    }
    OffloadSectionShadowVariables = New;
    CapSectionShadowVariables = NewCap;
  }
  if (IsVerboseMode())
    PROF_NOTE("Registering section shadow variable %d: %p\n",
              NumSectionShadowVariables, ptr);
  OffloadSectionShadowVariables[NumSectionShadowVariables++] = ptr;
}

static int ProcessDeviceOffloadPrf(void *DeviceOffloadPrf, int TUIndex) {
  void *HostOffloadPrf[8];

  if (IsVerboseMode())
    PROF_NOTE("HostOffloadPrf buffer size: %zu bytes\n",
              sizeof(HostOffloadPrf));

  if (hipMemcpy(HostOffloadPrf, DeviceOffloadPrf, sizeof(HostOffloadPrf),
                2 /*DToH*/) != 0) {
    PROF_ERR("%s\n", "Failed to copy offload prf structure from device");
    return -1;
  }

  void *DevCntsBegin = HostOffloadPrf[0];
  void *DevDataBegin = HostOffloadPrf[1];
  void *DevNamesBegin = HostOffloadPrf[2];
  void *DevUniformCntsBegin = HostOffloadPrf[3];
  void *DevCntsEnd = HostOffloadPrf[4];
  void *DevDataEnd = HostOffloadPrf[5];
  void *DevNamesEnd = HostOffloadPrf[6];
  void *DevUniformCntsEnd = HostOffloadPrf[7];

  if (IsVerboseMode()) {
    PROF_NOTE("%s", "Device Profile Pointers:\n");
    PROF_NOTE("  Counters:        %p - %p\n", DevCntsBegin, DevCntsEnd);
    PROF_NOTE("  Data:            %p - %p\n", DevDataBegin, DevDataEnd);
    PROF_NOTE("  Names:           %p - %p\n", DevNamesBegin, DevNamesEnd);
    PROF_NOTE("  UniformCounters: %p - %p\n", DevUniformCntsBegin,
              DevUniformCntsEnd);
  }

  size_t CountersSize = (char *)DevCntsEnd - (char *)DevCntsBegin;
  size_t DataSize = (char *)DevDataEnd - (char *)DevDataBegin;
  size_t NamesSize = (char *)DevNamesEnd - (char *)DevNamesBegin;
  size_t UniformCountersSize =
      (char *)DevUniformCntsEnd - (char *)DevUniformCntsBegin;

  if (IsVerboseMode()) {
    PROF_NOTE("Section sizes: Counters=%zu, Data=%zu, Names=%zu, "
              "UniformCounters=%zu\n",
              CountersSize, DataSize, NamesSize, UniformCountersSize);
  }

  if (CountersSize == 0 || DataSize == 0) {
    if (IsVerboseMode())
      PROF_NOTE("%s\n", "Counters or Data section has zero size. No profile "
                        "data to collect.");
    return 0;
  }

  // Pre-register device section symbols with CLR memory tracking.
  // This makes the section base pointers (and sub-pointers) safe for hipMemcpy.
  if (IsVerboseMode())
    PROF_NOTE("Pre-registering %d section symbols\n",
              NumSectionShadowVariables);
  for (int i = 0; i < NumSectionShadowVariables; ++i) {
    void *DevPtr = NULL;
    (void)hipGetSymbolAddress(&DevPtr, OffloadSectionShadowVariables[i]);
  }

  int ret = -1;

  // Allocate host memory for the device sections
  char *HostCountersBegin = (char *)malloc(CountersSize);
  char *HostDataBegin = (char *)malloc(DataSize);
  char *HostNamesBegin = (char *)malloc(NamesSize);
  char *HostUniformCountersBegin =
      (UniformCountersSize > 0) ? (char *)malloc(UniformCountersSize) : NULL;

  if (!HostCountersBegin || !HostDataBegin ||
      (NamesSize > 0 && !HostNamesBegin) ||
      (UniformCountersSize > 0 && !HostUniformCountersBegin)) {
    PROF_ERR("%s\n", "Failed to allocate host memory for device sections");
    goto cleanup;
  }

  // Copy data from device to host using HIP.
  if (memcpyDeviceToHost(HostCountersBegin, DevCntsBegin, CountersSize) != 0 ||
      memcpyDeviceToHost(HostDataBegin, DevDataBegin, DataSize) != 0 ||
      (NamesSize > 0 &&
       memcpyDeviceToHost(HostNamesBegin, DevNamesBegin, NamesSize) != 0) ||
      (UniformCountersSize > 0 &&
       memcpyDeviceToHost(HostUniformCountersBegin, DevUniformCntsBegin,
                          UniformCountersSize) != 0)) {
    PROF_ERR("%s\n", "Failed to copy profile sections from device");
    goto cleanup;
  }

  if (IsVerboseMode())
    PROF_NOTE("Copied device sections: Counters=%zu, Data=%zu, Names=%zu, "
              "UniformCounters=%zu\n",
              CountersSize, DataSize, NamesSize, UniformCountersSize);

  if (IsVerboseMode() && UniformCountersSize > 0) {
    PROF_NOTE("Successfully copied %zu bytes of uniform counters from device\n",
              UniformCountersSize);
  }

  // Compute padding sizes for proper buffer layout
  // lprofWriteDataImpl computes CountersDelta = CountersBegin - DataBegin
  // We need to arrange our buffer so this matches the expected file layout
  const uint64_t NumData = DataSize / sizeof(__llvm_profile_data);
  const uint64_t NumBitmapBytes = 0;
  const uint64_t VTableSectionSize = 0;
  const uint64_t VNamesSize = 0;
  uint64_t PaddingBytesBeforeCounters, PaddingBytesAfterCounters,
      PaddingBytesAfterBitmapBytes, PaddingBytesAfterNames,
      PaddingBytesAfterVTable, PaddingBytesAfterVNames;

  if (__llvm_profile_get_padding_sizes_for_counters(
          DataSize, CountersSize, NumBitmapBytes, NamesSize, VTableSectionSize,
          VNamesSize, &PaddingBytesBeforeCounters, &PaddingBytesAfterCounters,
          &PaddingBytesAfterBitmapBytes, &PaddingBytesAfterNames,
          &PaddingBytesAfterVTable, &PaddingBytesAfterVNames) != 0) {
    PROF_ERR("%s\n", "Failed to get padding sizes");
    goto cleanup;
  }

  // Create contiguous buffer with layout: [Data][Padding][Counters][Names]
  // This ensures CountersBegin - DataBegin = DataSize +
  // PaddingBytesBeforeCounters
  size_t ContiguousBufferSize =
      DataSize + PaddingBytesBeforeCounters + CountersSize + NamesSize;
  char *ContiguousBuffer = (char *)malloc(ContiguousBufferSize);
  if (!ContiguousBuffer) {
    PROF_ERR("%s\n", "Failed to allocate contiguous buffer");
    goto cleanup;
  }
  memset(ContiguousBuffer, 0, ContiguousBufferSize);

  // Set up pointers into the contiguous buffer
  char *BufDataBegin = ContiguousBuffer;
  char *BufCountersBegin =
      ContiguousBuffer + DataSize + PaddingBytesBeforeCounters;
  char *BufNamesBegin = BufCountersBegin + CountersSize;

  // Copy data into contiguous buffer
  memcpy(BufDataBegin, HostDataBegin, DataSize);
  memcpy(BufCountersBegin, HostCountersBegin, CountersSize);
  memcpy(BufNamesBegin, HostNamesBegin, NamesSize);

  // Relocate CounterPtr in data records for file layout
  // CounterPtr is device-relative offset; we need to adjust for file layout
  // where Data section comes first, then Counters section
  __llvm_profile_data *RelocatedData = (__llvm_profile_data *)BufDataBegin;
  for (uint64_t i = 0; i < NumData; ++i) {
    if (RelocatedData[i].CounterPtr) {
      ptrdiff_t DeviceCounterPtrOffset = (ptrdiff_t)RelocatedData[i].CounterPtr;
      void *DeviceDataStructAddr =
          (char *)DevDataBegin + (i * sizeof(__llvm_profile_data));
      void *DeviceCountersAddr =
          (char *)DeviceDataStructAddr + DeviceCounterPtrOffset;
      ptrdiff_t OffsetIntoCountersSection =
          (char *)DeviceCountersAddr - (char *)DevCntsBegin;

      // New offset: from this data record to its counters in file layout
      // CountersDelta = BufCountersBegin - BufDataBegin = DataSize + Padding
      // CounterPtr = CountersDelta + OffsetIntoCounters - (i * sizeof)
      ptrdiff_t NewRelativeOffset = DataSize + PaddingBytesBeforeCounters +
                                    OffsetIntoCountersSection -
                                    (i * sizeof(__llvm_profile_data));
      memcpy((char *)RelocatedData + i * sizeof(__llvm_profile_data) +
                 offsetof(__llvm_profile_data, CounterPtr),
             &NewRelativeOffset, sizeof(NewRelativeOffset));
    }
    // Zero out unused fields
    memset((char *)RelocatedData + i * sizeof(__llvm_profile_data) +
               offsetof(__llvm_profile_data, BitmapPtr),
           0,
           sizeof(RelocatedData[i].BitmapPtr) +
               sizeof(RelocatedData[i].FunctionPointer) +
               sizeof(RelocatedData[i].Values));
  }

  // Build TU suffix string for filename
  char TUIndexStr[16] = "";
  if (TUIndex >= 0) {
    snprintf(TUIndexStr, sizeof(TUIndexStr), "%d", TUIndex);
  }

  // Use shared profile writing API
  const char *TargetTriple = "amdgcn-amd-amdhsa";
  ret = __llvm_write_custom_profile(
      TargetTriple, TUIndex >= 0 ? TUIndexStr : NULL,
      (__llvm_profile_data *)BufDataBegin,
      (__llvm_profile_data *)(BufDataBegin + DataSize), BufCountersBegin,
      BufCountersBegin + CountersSize, HostUniformCountersBegin,
      HostUniformCountersBegin ? HostUniformCountersBegin + UniformCountersSize
                               : NULL,
      BufNamesBegin, BufNamesBegin + NamesSize, NULL);

  free(ContiguousBuffer);

  if (ret != 0) {
    PROF_ERR("%s\n", "Failed to write device profile using shared API");
  } else if (IsVerboseMode()) {
    PROF_NOTE("%s\n", "Successfully wrote device profile using shared API");
  }

cleanup:
  free(HostCountersBegin);
  free(HostDataBegin);
  free(HostNamesBegin);
  free(HostUniformCountersBegin);
  return ret;
}

static int ProcessShadowVariable(void *ShadowVar, int TUIndex) {
  void *DeviceOffloadPrf = NULL;
  if (hipGetSymbolAddress(&DeviceOffloadPrf, ShadowVar) != 0) {
    PROF_WARN("Failed to get symbol address for shadow variable %p\n",
              ShadowVar);
    return -1;
  }
  return ProcessDeviceOffloadPrf(DeviceOffloadPrf, TUIndex);
}

/* Check if HIP runtime is available and loaded */
static int IsHipAvailable(void) {
  EnsureHipLoaded();
  return pHipMemcpy != NULL && pHipGetSymbolAddress != NULL;
}

/* -------------------------------------------------------------------------- */
/*  Collect device-side profile data                                          */
/* -------------------------------------------------------------------------- */

int __llvm_profile_hip_collect_device_data(void) {
  if (IsVerboseMode())
    PROF_NOTE("%s", "__llvm_profile_hip_collect_device_data called\n");

  /* Early return if no HIP profile data was registered */
  if (NumShadowVariables == 0 && NumDynamicModules == 0) {
    if (IsVerboseMode())
      PROF_NOTE("%s", "No HIP profile data registered, skipping collection\n");
    return 0;
  }

  /* Early return if HIP runtime is not available */
  if (!IsHipAvailable()) {
    if (IsVerboseMode())
      PROF_NOTE("%s", "HIP runtime not available, skipping collection\n");
    return 0;
  }

  int Ret = 0;

  /* Shadow variables (static-linked kernels) */
  /* Always use TU index for consistent naming
   * (profile.amdgcn-amd-amdhsa.0.profraw, etc.) */
  for (int i = 0; i < NumShadowVariables; ++i) {
    if (ProcessShadowVariable(OffloadShadowVariables[i], i) != 0)
      Ret = -1;
  }

  /* Dynamically-loaded modules — warn about any unprocessed TUs */
  for (int i = 0; i < NumDynamicModules; ++i) {
    OffloadDynamicModuleInfo *MI = &DynamicModules[i];
    for (int t = 0; t < MI->NumTUs; ++t) {
      if (!MI->TUs[t].Processed) {
        PROF_WARN("Dynamic module %p TU %d was not processed before exit\n",
                  MI->ModulePtr, t);
        Ret = -1;
      }
    }
  }

  return Ret;
}
