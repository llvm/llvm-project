//===- InstrProfilingPlatformROCm.cpp - Profile data ROCm platform -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

extern "C" {
#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include "InstrProfilingPort.h"
}

#include "interception/interception.h"
// C library headers (not <cstdio> etc.): clang_rt.profile is built with
// -nostdinc++ and avoids the C++ standard library (see profile/CMakeLists.txt).
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int processDeviceOffloadPrf(void *DeviceOffloadPrf, int TUIndex,
                                   const char *Target);

static int isVerboseMode() {
  static int IsVerbose = -1;
  if (IsVerbose == -1)
    IsVerbose = getenv("LLVM_PROFILE_VERBOSE") != nullptr;
  return IsVerbose;
}

/* -------------------------------------------------------------------------- */
/*  Dynamic loading of HIP runtime symbols                                   */
/* -------------------------------------------------------------------------- */

typedef int (*hipGetSymbolAddressTy)(void **, const void *);
typedef int (*hipMemcpyTy)(void *, const void *, size_t, int);
typedef int (*hipModuleGetGlobalTy)(void **, size_t *, void *, const char *);
typedef int (*hipGetDeviceCountTy)(int *);
typedef int (*hipGetDeviceTy)(int *);
typedef int (*hipSetDeviceTy)(int);

/* hipDeviceProp_t layout for HIP 6.x+ (R0600).
 * We only need gcnArchName at offset 1160. Pad to 4096 to safely
 * accommodate future struct growth without recompilation. */
typedef struct {
  char padding[1160];
  char gcnArchName[256];
  char tail_padding[2680];
} HipDevicePropMinimal;
typedef int (*hipGetDevicePropertiesTy)(HipDevicePropMinimal *, int);

static hipGetSymbolAddressTy pHipGetSymbolAddress = nullptr;
static hipMemcpyTy pHipMemcpy = nullptr;
static hipModuleGetGlobalTy pHipModuleGetGlobal = nullptr;
static hipGetDeviceCountTy pHipGetDeviceCount = nullptr;
static hipGetDeviceTy pHipGetDevice = nullptr;
static hipSetDeviceTy pHipSetDevice = nullptr;
static hipGetDevicePropertiesTy pHipGetDeviceProperties = nullptr;

#define MAX_DEVICES 16
static int NumDevices = 0;
static char DeviceArchNames[MAX_DEVICES][256];

/* -------------------------------------------------------------------------- */
/*  Device-to-host copies                                                     */
/*  Keep HIP-only to avoid an HSA dependency.                                 */
/* -------------------------------------------------------------------------- */

static void ensureHipLoaded(void) {
  static int Initialized = 0;
  if (Initialized)
    return;
  Initialized = 1;

  if (!__interception::DynamicLoaderAvailable()) {
    if (isVerboseMode())
      PROF_NOTE("%s", "Dynamic library loading not available - "
                      "HIP profiling disabled\n");
    return;
  }

#ifdef _WIN32
  static const char HipLibName[] = "amdhip64.dll";
#else
  static const char HipLibName[] = "libamdhip64.so";
#endif

  void *Handle = __interception::OpenLibrary(HipLibName);
  if (!Handle)
    return;

  pHipGetSymbolAddress = (hipGetSymbolAddressTy)__interception::LookupSymbol(
      Handle, "hipGetSymbolAddress");
  pHipMemcpy = (hipMemcpyTy)__interception::LookupSymbol(Handle, "hipMemcpy");
  pHipModuleGetGlobal = (hipModuleGetGlobalTy)__interception::LookupSymbol(
      Handle, "hipModuleGetGlobal");
  pHipGetDeviceCount = (hipGetDeviceCountTy)__interception::LookupSymbol(
      Handle, "hipGetDeviceCount");
  pHipGetDevice =
      (hipGetDeviceTy)__interception::LookupSymbol(Handle, "hipGetDevice");
  pHipSetDevice =
      (hipSetDeviceTy)__interception::LookupSymbol(Handle, "hipSetDevice");
  pHipGetDeviceProperties =
      (hipGetDevicePropertiesTy)__interception::LookupSymbol(
          Handle, "hipGetDevicePropertiesR0600");
  if (!pHipGetDeviceProperties)
    pHipGetDeviceProperties =
        (hipGetDevicePropertiesTy)__interception::LookupSymbol(
            Handle, "hipGetDeviceProperties");

  if (pHipGetDeviceCount && pHipGetDeviceProperties) {
    int Count = 0;
    if (pHipGetDeviceCount(&Count) == 0) {
      if (Count > MAX_DEVICES)
        Count = MAX_DEVICES;
      HipDevicePropMinimal Prop;
      for (int i = 0; i < Count; ++i) {
        __builtin_memset(&Prop, 0, sizeof(Prop));
        if (pHipGetDeviceProperties(&Prop, i) == 0) {
          strncpy(DeviceArchNames[i], Prop.gcnArchName,
                  sizeof(DeviceArchNames[i]) - 1);
          DeviceArchNames[i][sizeof(DeviceArchNames[i]) - 1] = '\0';
          if (isVerboseMode())
            PROF_NOTE("Device %d arch: %s\n", i, DeviceArchNames[i]);
        }
      }
      NumDevices = Count;
    }
  }
}

/* -------------------------------------------------------------------------- */
/*  Public wrappers that forward to the loaded HIP symbols                   */
/* -------------------------------------------------------------------------- */

static int hipGetSymbolAddress(void **devPtr, const void *symbol) {
  ensureHipLoaded();
  return pHipGetSymbolAddress ? pHipGetSymbolAddress(devPtr, symbol) : -1;
}

static int hipMemcpy(void *dest, const void *src, size_t len,
                     int kind /*2=DToH*/) {
  ensureHipLoaded();
  return pHipMemcpy ? pHipMemcpy(dest, src, len, kind) : -1;
}

/* Copy from device to host using HIP.
 * This requires that the device section symbols are registered with CLR,
 * otherwise hipMemcpy may attempt a CPU path and crash. */
static int memcpyDeviceToHost(void *Dst, const void *Src, size_t Size) {
  return hipMemcpy(Dst, Src, Size, 2 /* DToH */);
}

static int hipModuleGetGlobal(void **DevPtr, size_t *Bytes, void *Module,
                              const char *Name) {
  ensureHipLoaded();
  return pHipModuleGetGlobal ? pHipModuleGetGlobal(DevPtr, Bytes, Module, Name)
                             : -1;
}

static int hipGetDevice(int *DeviceId) {
  ensureHipLoaded();
  return pHipGetDevice ? pHipGetDevice(DeviceId) : -1;
}

static int hipSetDevice(int DeviceId) {
  ensureHipLoaded();
  return pHipSetDevice ? pHipSetDevice(DeviceId) : -1;
}

static const char *getDeviceArchName(int DeviceId) {
  if (DeviceId < 0 || DeviceId >= NumDevices || !DeviceArchNames[DeviceId][0])
    return "amdgpu";
  return DeviceArchNames[DeviceId];
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

static OffloadDynamicModuleInfo *DynamicModules = nullptr;
static int NumDynamicModules = 0;
static int CapDynamicModules = 0;

/* -------------------------------------------------------------------------- */
/*  ELF symbol enumeration                                                    */
/*                                                                            */
/*  AMDGPU code objects are always ELF.  We use manual parsing because this   */
/*  is compiler-rt (standalone C runtime) and cannot link against LLVM's C++  */
/*  Support libraries.                                                        */
/* -------------------------------------------------------------------------- */

#if __has_include(<elf.h>)
#include <elf.h>

/* Callback invoked for every matching symbol name found in the ELF image.
 * Return 0 to continue iteration, non-zero to stop. */
typedef int (*SymbolCallback)(const char *Name, void *UserData);

/* If Image is a clang offload bundle (__CLANG_OFFLOAD_BUNDLE__), find the
 * first embedded code object that is a valid ELF and return a pointer to it.
 * Otherwise return Image unchanged. Returns nullptr if no ELF is found. */
static const void *unwrapOffloadBundle(const void *Image) {
  static const char BundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
  if (memcmp(Image, BundleMagic, sizeof(BundleMagic) - 1) != 0)
    return Image; /* Not a bundle, return as-is. */

  const char *Buf = (const char *)Image;
  uint64_t NumEntries;
  __builtin_memcpy(&NumEntries, Buf + sizeof(BundleMagic) - 1,
                   sizeof(uint64_t));

  /* Walk the entry table (starts at offset 32). */
  const char *Cursor = Buf + 32;
  for (uint64_t I = 0; I < NumEntries; ++I) {
    uint64_t EntryOffset, EntrySize, IDSize;
    __builtin_memcpy(&EntryOffset, Cursor, sizeof(EntryOffset));
    Cursor += sizeof(EntryOffset);
    __builtin_memcpy(&EntrySize, Cursor, sizeof(EntrySize));
    Cursor += sizeof(EntrySize);
    __builtin_memcpy(&IDSize, Cursor, sizeof(IDSize));
    Cursor += sizeof(IDSize);
    /* Skip the entry ID string. */
    Cursor += IDSize;

    /* Check if this entry contains an ELF. */
    if (EntrySize >= sizeof(Elf64_Ehdr)) {
      const Elf64_Ehdr *E = (const Elf64_Ehdr *)(Buf + EntryOffset);
      if (E->e_ident[EI_MAG0] == ELFMAG0 && E->e_ident[EI_MAG1] == ELFMAG1 &&
          E->e_ident[EI_MAG2] == ELFMAG2 && E->e_ident[EI_MAG3] == ELFMAG3) {
        return (const void *)(Buf + EntryOffset);
      }
    }
  }

  PROF_WARN("%s", "offload bundle contains no valid ELF entries\n");
  return nullptr;
}

/* Parse an AMDGPU code-object ELF and invoke CB for every global symbol whose
 * name starts with PREFIX.  Image may be nullptr (e.g. hipModuleLoad from file)
 * or a clang offload bundle containing an ELF;
 * in that case the function unwraps the bundle first. */
static void enumerateElfSymbols(const void *Image, const char *Prefix,
                                SymbolCallback CB, void *UserData) {
  if (!Image)
    return;

  /* Handle clang offload bundle wrapping. */
  Image = unwrapOffloadBundle(Image);
  if (!Image)
    return;

  const Elf64_Ehdr *Ehdr = (const Elf64_Ehdr *)Image;
  if (Ehdr->e_ident[EI_MAG0] != ELFMAG0 || Ehdr->e_ident[EI_MAG1] != ELFMAG1 ||
      Ehdr->e_ident[EI_MAG2] != ELFMAG2 || Ehdr->e_ident[EI_MAG3] != ELFMAG3) {
    if (isVerboseMode())
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
static int registerPrfSymbol(const char *Name, void *UserData) {
  EnumState *S = (EnumState *)UserData;
  OffloadDynamicModuleInfo *MI = S->ModInfo;

  /* Look up the per-TU pointer variable, then dereference to get the
   * address of __llvm_profile_sections. */
  void *DevicePtrVar = nullptr;
  size_t Bytes = 0;
  if (hipModuleGetGlobal(&DevicePtrVar, &Bytes, S->Module, Name) != 0) {
    PROF_WARN("failed to get symbol %s for module %p\n", Name, S->Module);
    return 0; /* continue */
  }
  void *DeviceVar = nullptr;
  if (hipMemcpy(&DeviceVar, DevicePtrVar, sizeof(void *), 2 /*DToH*/) != 0) {
    PROF_WARN("failed to read sections pointer for %s\n", Name);
    return 0;
  }

  /* Grow TU array if needed. */
  if (MI->NumTUs >= MI->CapTUs) {
    int NewCap = MI->CapTUs ? MI->CapTUs * 2 : 4;
    OffloadDynamicTUInfo *New = (OffloadDynamicTUInfo *)realloc(
        MI->TUs, NewCap * sizeof(OffloadDynamicTUInfo));
    if (!New) {
      PROF_ERR("%s\n", "failed to grow TU array");
      return 0;
    }
    MI->TUs = New;
    MI->CapTUs = NewCap;
  }
  OffloadDynamicTUInfo *TU = &MI->TUs[MI->NumTUs++];
  TU->DeviceVar = DeviceVar;
  TU->Processed = 0;

  (void)Name; /* CUID suffix available for future per-TU section lookup */

  return 0; /* continue enumeration */
}

#endif /* __has_include(<elf.h>) */

/* -------------------------------------------------------------------------- */
/*  Registration / un-registration helpers                                   */
/* -------------------------------------------------------------------------- */

extern "C" void
__llvm_profile_offload_register_dynamic_module(int ModuleLoadRc, void **Ptr,
                                               const void *Image) {
  if (isVerboseMode())
    PROF_NOTE("Registering loaded module %d: rc=%d, module=%p, image=%p\n",
              NumDynamicModules, ModuleLoadRc, *Ptr, Image);

  if (ModuleLoadRc)
    return;

  if (NumDynamicModules >= CapDynamicModules) {
    int NewCap = CapDynamicModules ? CapDynamicModules * 2 : 64;
    OffloadDynamicModuleInfo *New = (OffloadDynamicModuleInfo *)realloc(
        DynamicModules, NewCap * sizeof(OffloadDynamicModuleInfo));
    if (!New)
      return;
    DynamicModules = New;
    CapDynamicModules = NewCap;
  }

  OffloadDynamicModuleInfo *MI = &DynamicModules[NumDynamicModules++];
  MI->ModulePtr = *Ptr;
  MI->TUs = nullptr;
  MI->NumTUs = 0;
  MI->CapTUs = 0;

  /* Enumerate all __llvm_offload_prf_<CUID> symbols in the ELF image.
   * For each one, look it up via hipModuleGetGlobal (which also registers
   * the device address with CLR for later hipMemcpy) and store the entry.
   *
   * ELF parsing requires <elf.h>.  On platforms without it, dynamic module
   * profiling is not yet supported. */
#if __has_include(<elf.h>)
  EnumState State = {*Ptr, MI};
  enumerateElfSymbols(Image, "__llvm_offload_prf_", registerPrfSymbol, &State);
#else
  (void)Image;
  if (isVerboseMode())
    PROF_NOTE("%s",
              "Dynamic module profiling not supported on this platform\n");
#endif

  if (MI->NumTUs == 0) {
    PROF_WARN("no __llvm_offload_prf_* symbols found in module %p\n", *Ptr);
  } else if (isVerboseMode()) {
    PROF_NOTE("Module %p: registered %d TU(s)\n", *Ptr, MI->NumTUs);
  }
}

extern "C" void __llvm_profile_offload_unregister_dynamic_module(void *Ptr) {
  for (int i = 0; i < NumDynamicModules; ++i) {
    OffloadDynamicModuleInfo *MI = &DynamicModules[i];

    if (MI->ModulePtr != Ptr)
      continue;

    if (isVerboseMode())
      PROF_NOTE("Unregistering module %p (%d TUs)\n", MI->ModulePtr,
                MI->NumTUs);

    /* Process every TU in this module. */
    for (int t = 0; t < MI->NumTUs; ++t) {
      OffloadDynamicTUInfo *TU = &MI->TUs[t];
      if (TU->Processed) {
        if (isVerboseMode())
          PROF_NOTE("Module %p TU %d already processed, skipping\n", Ptr, t);
        continue;
      }
      /* Use a globally unique index as TU index for the output filename. */
      int TUIndex = i * 1000 + t;
      if (TU->DeviceVar) {
        int CurDev = 0;
        hipGetDevice(&CurDev);
        const char *ArchName = getDeviceArchName(CurDev);
        if (processDeviceOffloadPrf(TU->DeviceVar, TUIndex, ArchName) == 0)
          TU->Processed = 1;
        else
          PROF_WARN("failed to process profile data for module %p TU %d\n", Ptr,
                    t);
      }
    }
    return;
  }

  if (isVerboseMode())
    PROF_WARN("unregister called for unknown module %p\n", Ptr);
}

/* Grow a void* array, doubling capacity (or starting at InitCap). */
static int growPtrArray(void ***Arr, int *Num, int *Cap, int InitCap) {
  if (*Num < *Cap)
    return 0;
  int NewCap = *Cap ? *Cap * 2 : InitCap;
  void **New = (void **)realloc(*Arr, NewCap * sizeof(void *));
  if (!New)
    return -1;
  *Arr = New;
  *Cap = NewCap;
  return 0;
}

static void **OffloadShadowVariables = nullptr;
static int NumShadowVariables = 0;
static int CapShadowVariables = 0;

extern "C" void __llvm_profile_offload_register_shadow_variable(void *ptr) {
  if (growPtrArray(&OffloadShadowVariables, &NumShadowVariables,
                   &CapShadowVariables, 64))
    return;
  OffloadShadowVariables[NumShadowVariables++] = ptr;
}

static void **OffloadSectionShadowVariables = nullptr;
static int NumSectionShadowVariables = 0;
static int CapSectionShadowVariables = 0;

extern "C" void
__llvm_profile_offload_register_section_shadow_variable(void *ptr) {
  if (growPtrArray(&OffloadSectionShadowVariables, &NumSectionShadowVariables,
                   &CapSectionShadowVariables, 64))
    return;
  OffloadSectionShadowVariables[NumSectionShadowVariables++] = ptr;
}

// Free host-side copies of device sections on error or success. Factored out
// so we can return early: C++ forbids goto past initializations of automatic
// locals declared later in this function (e.g. const uint64_t NumData).
// Callers pass nullptr for reused (cached) sections so we only free malloc'd
// buffers; free(nullptr) is a no-op (C/C++).
static void freeCopiedHostSections(char *HostCountersBegin, char *HostDataBegin,
                                   char *HostNamesBegin) {
  free(HostCountersBegin);
  free(HostDataBegin);
  free(HostNamesBegin);
}

namespace {

struct CopiedHostSectionsCleanup {
  char *Counters;
  char *Data;
  char *Names;
  int CntsReused;
  int DataReused;
  int NamesReused;

  CopiedHostSectionsCleanup(char *C, char *D, char *N, int CR, int DR, int NR)
      : Counters(C), Data(D), Names(N), CntsReused(CR), DataReused(DR),
        NamesReused(NR) {}

  ~CopiedHostSectionsCleanup() {
    freeCopiedHostSections(CntsReused ? nullptr : Counters,
                           DataReused ? nullptr : Data,
                           NamesReused ? nullptr : Names);
  }

  CopiedHostSectionsCleanup(const CopiedHostSectionsCleanup &) = delete;
  CopiedHostSectionsCleanup &
  operator=(const CopiedHostSectionsCleanup &) = delete;
};

struct MallocBufferCleanup {
  void *Ptr;
  explicit MallocBufferCleanup(void *P) : Ptr(P) {}
  ~MallocBufferCleanup() { free(Ptr); }
  MallocBufferCleanup(const MallocBufferCleanup &) = delete;
  MallocBufferCleanup &operator=(const MallocBufferCleanup &) = delete;
  char *get() const { return static_cast<char *>(Ptr); }
};

} // namespace

static int processDeviceOffloadPrf(void *DeviceOffloadPrf, int TUIndex,
                                   const char *Target) {
  __llvm_profile_gpu_sections HostSections;

  if (hipMemcpy(&HostSections, DeviceOffloadPrf, sizeof(HostSections),
                2 /*DToH*/) != 0) {
    PROF_ERR("%s\n", "failed to copy offload prf structure from device");
    return -1;
  }

  const void *DevCntsBegin = HostSections.CountersStart;
  const void *DevDataBegin = HostSections.DataStart;
  const void *DevNamesBegin = HostSections.NamesStart;
  const void *DevCntsEnd = HostSections.CountersStop;
  const void *DevDataEnd = HostSections.DataStop;
  const void *DevNamesEnd = HostSections.NamesStop;

  size_t CountersSize = (const char *)DevCntsEnd - (const char *)DevCntsBegin;
  size_t DataSize = (const char *)DevDataEnd - (const char *)DevDataBegin;
  size_t NamesSize = (const char *)DevNamesEnd - (const char *)DevNamesBegin;

  if (isVerboseMode())
    PROF_NOTE("Section pointers: Cnts=[%p,%p]=%zu Data=[%p,%p]=%zu "
              "Names=[%p,%p]=%zu\n",
              DevCntsBegin, DevCntsEnd, CountersSize, DevDataBegin, DevDataEnd,
              DataSize, DevNamesBegin, DevNamesEnd, NamesSize);

  if (CountersSize == 0 || DataSize == 0)
    return 0;

  int ret = -1;
  int NamesReused = 0, CntsReused = 0, DataReused = 0;

  char *HostDataBegin = nullptr;
  char *HostCountersBegin = nullptr;
  char *HostNamesBegin = nullptr;

  /* Sections using linker-defined __start_/__stop_ bounds are shared across
     TU structs in RDC mode. Deduplicate by caching the last copied range. */
  static const void *CachedDevNamesBegin = nullptr;
  static char *CachedHostNames = nullptr;
  static size_t CachedNamesSize = 0;

  static const void *CachedDevCntsBegin = nullptr;
  static char *CachedHostCnts = nullptr;
  static size_t CachedCntsSize = 0;

  static const void *CachedDevDataBegin = nullptr;
  static char *CachedHostData = nullptr;
  static size_t CachedDataSize = 0;

  if (CountersSize > 0 && DevCntsBegin == CachedDevCntsBegin &&
      CountersSize == CachedCntsSize) {
    HostCountersBegin = CachedHostCnts;
    CntsReused = 1;
    if (isVerboseMode())
      PROF_NOTE("Reusing cached counters section (%zu bytes)\n", CountersSize);
  } else if (CountersSize > 0) {
    HostCountersBegin = (char *)malloc(CountersSize);
  }

  if (DataSize > 0 && DevDataBegin == CachedDevDataBegin &&
      DataSize == CachedDataSize) {
    HostDataBegin = CachedHostData;
    DataReused = 1;
    if (isVerboseMode())
      PROF_NOTE("Reusing cached data section (%zu bytes)\n", DataSize);
  } else if (DataSize > 0) {
    HostDataBegin = (char *)malloc(DataSize);
  }

  if (NamesSize > 0 && DevNamesBegin == CachedDevNamesBegin &&
      NamesSize == CachedNamesSize) {
    HostNamesBegin = CachedHostNames;
    NamesReused = 1;
    if (isVerboseMode())
      PROF_NOTE("Reusing cached names section (%zu bytes)\n", NamesSize);
  } else if (NamesSize > 0) {
    HostNamesBegin = (char *)malloc(NamesSize);
  }

  // On failure before the contiguous buffer exists, free host copies and
  // return. Do not use goto cleanup: later locals make that ill-formed C++.
  if ((DataSize > 0 && !HostDataBegin) ||
      (CountersSize > 0 && !HostCountersBegin) ||
      (NamesSize > 0 && !HostNamesBegin)) {
    PROF_ERR("%s\n", "failed to allocate host memory for device sections");
    freeCopiedHostSections(CntsReused ? nullptr : HostCountersBegin,
                           DataReused ? nullptr : HostDataBegin,
                           NamesReused ? nullptr : HostNamesBegin);
    return -1;
  }

  CopiedHostSectionsCleanup HostCopies(HostCountersBegin, HostDataBegin,
                                       HostNamesBegin, CntsReused, DataReused,
                                       NamesReused);

  if ((DataSize > 0 && !DataReused &&
       memcpyDeviceToHost(HostDataBegin, DevDataBegin, DataSize) != 0) ||
      (CountersSize > 0 && !CntsReused &&
       memcpyDeviceToHost(HostCountersBegin, DevCntsBegin, CountersSize) !=
           0) ||
      (NamesSize > 0 && !NamesReused &&
       memcpyDeviceToHost(HostNamesBegin, DevNamesBegin, NamesSize) != 0)) {
    PROF_ERR("%s\n", "failed to copy profile sections from device");
    return -1;
  }

  if (!CntsReused && CountersSize > 0) {
    CachedDevCntsBegin = DevCntsBegin;
    CachedHostCnts = HostCountersBegin;
    CachedCntsSize = CountersSize;
  }
  if (!DataReused && DataSize > 0) {
    CachedDevDataBegin = DevDataBegin;
    CachedHostData = HostDataBegin;
    CachedDataSize = DataSize;
  }
  if (!NamesReused && NamesSize > 0) {
    CachedDevNamesBegin = DevNamesBegin;
    CachedHostNames = HostNamesBegin;
    CachedNamesSize = NamesSize;
  }

  if (isVerboseMode())
    PROF_NOTE("Copied device sections: Counters=%zu, Data=%zu, Names=%zu\n",
              CountersSize, DataSize, NamesSize);

  // Arrange buffer as [Data][Padding][Counters][Names] to match the layout
  // expected by lprofWriteDataImpl (CountersDelta = CountersBegin - DataBegin).
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
    PROF_ERR("%s\n", "failed to get padding sizes");
    return -1;
  }

  size_t ContiguousBufferSize =
      DataSize + PaddingBytesBeforeCounters + CountersSize + NamesSize;
  MallocBufferCleanup ContiguousBuf(malloc(ContiguousBufferSize));
  if (!ContiguousBuf.get()) {
    PROF_ERR("%s\n", "failed to allocate contiguous buffer");
    return -1;
  }
  char *ContiguousBuffer = ContiguousBuf.get();
  __builtin_memset(ContiguousBuffer, 0, ContiguousBufferSize);

  char *BufDataBegin = ContiguousBuffer;
  char *BufCountersBegin =
      ContiguousBuffer + DataSize + PaddingBytesBeforeCounters;
  char *BufNamesBegin = BufCountersBegin + CountersSize;

  __builtin_memcpy(BufDataBegin, HostDataBegin, DataSize);
  __builtin_memcpy(BufCountersBegin, HostCountersBegin, CountersSize);
  __builtin_memcpy(BufNamesBegin, HostNamesBegin, NamesSize);

  // Relocate CounterPtr in data records for file layout.
  // CounterPtr is device-relative offset; adjust for file layout where
  // Data section comes first, then Counters section.
  __llvm_profile_data *RelocatedData = (__llvm_profile_data *)BufDataBegin;
  for (uint64_t i = 0; i < NumData; ++i) {
    if (RelocatedData[i].CounterPtr) {
      ptrdiff_t DeviceCounterPtrOffset = (ptrdiff_t)RelocatedData[i].CounterPtr;
      const char *DeviceDataStructAddr =
          (const char *)DevDataBegin + (i * sizeof(__llvm_profile_data));
      const char *DeviceCountersAddr =
          DeviceDataStructAddr + DeviceCounterPtrOffset;
      ptrdiff_t OffsetIntoCountersSection =
          DeviceCountersAddr - (const char *)DevCntsBegin;

      ptrdiff_t NewRelativeOffset = DataSize + PaddingBytesBeforeCounters +
                                    OffsetIntoCountersSection -
                                    (i * sizeof(__llvm_profile_data));
      __builtin_memcpy((char *)RelocatedData + i * sizeof(__llvm_profile_data) +
                           offsetof(__llvm_profile_data, CounterPtr),
                       &NewRelativeOffset, sizeof(NewRelativeOffset));
    }
    __builtin_memset((char *)RelocatedData + i * sizeof(__llvm_profile_data) +
                         offsetof(__llvm_profile_data, BitmapPtr),
                     0,
                     sizeof(RelocatedData[i].BitmapPtr) +
                         sizeof(RelocatedData[i].FunctionPointer) +
                         sizeof(RelocatedData[i].Values));
  }

  char TUIndexStr[16];
  snprintf(TUIndexStr, sizeof(TUIndexStr), "%d", TUIndex);

  ret = __llvm_write_custom_profile(
      Target, (__llvm_profile_data *)BufDataBegin,
      (__llvm_profile_data *)(BufDataBegin + DataSize), BufCountersBegin,
      BufCountersBegin + CountersSize, BufNamesBegin, BufNamesBegin + NamesSize,
      nullptr);

  if (ret != 0) {
    PROF_ERR("%s\n", "failed to write device profile using shared API");
  } else if (isVerboseMode()) {
    PROF_NOTE("%s\n", "Successfully wrote device profile using shared API");
  }

  return ret;
}

static int processShadowVariable(void *ShadowVar, int TUIndex,
                                 const char *Target) {
  void *DevicePtrVar = nullptr;
  if (hipGetSymbolAddress(&DevicePtrVar, ShadowVar) != 0) {
    PROF_WARN("failed to get symbol address for shadow variable %p\n",
              ShadowVar);
    return -1;
  }
  // The shadow variable is a pointer to __llvm_profile_sections (defined
  // in the GPU profile runtime). Dereference to get the struct address.
  void *DeviceOffloadPrf = nullptr;
  if (hipMemcpy(&DeviceOffloadPrf, DevicePtrVar, sizeof(void *), 2 /*DToH*/) !=
      0) {
    PROF_WARN("failed to read sections pointer from shadow variable %p\n",
              ShadowVar);
    return -1;
  }
  return processDeviceOffloadPrf(DeviceOffloadPrf, TUIndex, Target);
}

/* Check if HIP runtime is available and loaded */
static int isHipAvailable(void) {
  ensureHipLoaded();
  return pHipMemcpy != nullptr && pHipGetSymbolAddress != nullptr;
}

/* -------------------------------------------------------------------------- */
/*  Collect device-side profile data                                          */
/* -------------------------------------------------------------------------- */

extern "C" int __llvm_profile_hip_collect_device_data(void) {
  if (NumShadowVariables == 0 && NumDynamicModules == 0)
    return 0;

  if (!isHipAvailable())
    return 0;

  int Ret = 0;

  /* Shadow variables (static-linked kernels).
   * Iterate over all devices to collect profile data from each GPU. */
  if (NumShadowVariables > 0) {
    int OrigDevice = -1;
    hipGetDevice(&OrigDevice);

    for (int Dev = 0; Dev < NumDevices; ++Dev) {
      if (hipSetDevice(Dev) != 0) {
        if (isVerboseMode())
          PROF_NOTE("Failed to set device %d, skipping\n", Dev);
        continue;
      }
      const char *ArchName = getDeviceArchName(Dev);
      if (isVerboseMode())
        PROF_NOTE("Collecting static profile data from device %d (%s)\n", Dev,
                  ArchName);
      for (int i = 0; i < NumShadowVariables; ++i) {
        if (processShadowVariable(OffloadShadowVariables[i], i, ArchName) != 0)
          Ret = -1;
      }
    }

    if (OrigDevice >= 0)
      hipSetDevice(OrigDevice);
  }

  /* Dynamically-loaded modules — warn about any unprocessed TUs */
  for (int i = 0; i < NumDynamicModules; ++i) {
    OffloadDynamicModuleInfo *MI = &DynamicModules[i];
    for (int t = 0; t < MI->NumTUs; ++t) {
      if (!MI->TUs[t].Processed) {
        PROF_WARN("dynamic module %p TU %d was not processed before exit\n",
                  MI->ModulePtr, t);
        Ret = -1;
      }
    }
  }

  return Ret;
}
