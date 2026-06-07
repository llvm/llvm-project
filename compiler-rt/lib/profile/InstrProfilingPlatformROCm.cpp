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

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <pthread.h>
#endif

/* Serialize one-time HIP loader resolution and DynamicModules mutations.
 * Inline to avoid a sanitizer_common dependency. */
#ifdef _WIN32
static INIT_ONCE HipLoadedOnce = INIT_ONCE_STATIC_INIT;
static CRITICAL_SECTION DynamicModulesLock;
static INIT_ONCE DynamicModulesLockInit = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK initDynamicModulesLockCb(PINIT_ONCE, PVOID, PVOID *) {
  InitializeCriticalSection(&DynamicModulesLock);
  return TRUE;
}
static void lockDynamicModules(void) {
  InitOnceExecuteOnce(&DynamicModulesLockInit, initDynamicModulesLockCb, NULL,
                      NULL);
  EnterCriticalSection(&DynamicModulesLock);
}
static void unlockDynamicModules(void) {
  LeaveCriticalSection(&DynamicModulesLock);
}
#else
static pthread_once_t HipLoadedOnce = PTHREAD_ONCE_INIT;
static pthread_mutex_t DynamicModulesLock = PTHREAD_MUTEX_INITIALIZER;
static void lockDynamicModules(void) {
  pthread_mutex_lock(&DynamicModulesLock);
}
static void unlockDynamicModules(void) {
  pthread_mutex_unlock(&DynamicModulesLock);
}
#endif

struct OffloadSectionShadowGroup;
static int processDeviceOffloadPrf(void *DeviceOffloadPrf, const char *Target,
                                   const OffloadSectionShadowGroup *Sections);

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
typedef int (*hipGetSymbolSizeTy)(size_t *, const void *);
typedef int (*hipMemcpyTy)(void *, const void *, size_t, int);
typedef int (*hipModuleGetGlobalTy)(void **, size_t *, void *, const char *);
typedef int (*hipGetDeviceCountTy)(int *);
typedef int (*hipGetDeviceTy)(int *);
typedef int (*hipSetDeviceTy)(int);

/* Minimal hipDeviceProp_t (HIP 6.x R0600): only gcnArchName at offset 1160
 * is read. Padded to 4096 to tolerate ABI growth. */
typedef struct {
  char padding[1160];
  char gcnArchName[256];
  char tail_padding[2680];
} HipDevicePropMinimal;
typedef int (*hipGetDevicePropertiesTy)(HipDevicePropMinimal *, int);

static hipGetSymbolAddressTy pHipGetSymbolAddress = nullptr;
static hipGetSymbolSizeTy pHipGetSymbolSize = nullptr;
static hipMemcpyTy pHipMemcpy = nullptr;
static hipModuleGetGlobalTy pHipModuleGetGlobal = nullptr;
static hipGetDeviceCountTy pHipGetDeviceCount = nullptr;
static hipGetDeviceTy pHipGetDevice = nullptr;
static hipSetDeviceTy pHipSetDevice = nullptr;
static hipGetDevicePropertiesTy pHipGetDeviceProperties = nullptr;

static int NumDevices = 0;
/* 256 matches hipDeviceProp_t::gcnArchName, the source field width. */
static char (*DeviceArchNames)[256] = nullptr;

/* -------------------------------------------------------------------------- */
/*  Device-to-host copies                                                     */
/*  Keep HIP-only to avoid an HSA dependency.                                 */
/* -------------------------------------------------------------------------- */

static void doEnsureHipLoaded(void) {
  if (!__interception::DynamicLoaderAvailable()) {
    if (isVerboseMode())
      PROF_NOTE("%s", "Dynamic library loading not available - "
                      "HIP profiling disabled\n");
    return;
  }

#ifdef _WIN32
  const char *HipLibName = "amdhip64_7.dll";
#else
  const char *HipLibName = "libamdhip64.so";
#endif

  void *Handle = __interception::OpenLibrary(HipLibName);
#ifdef _WIN32
  if (!Handle) {
    HipLibName = "amdhip64.dll";
    Handle = __interception::OpenLibrary(HipLibName);
  }
#endif
  if (!Handle)
    return;

  pHipGetSymbolAddress = (hipGetSymbolAddressTy)__interception::LookupSymbol(
      Handle, "hipGetSymbolAddress");
  pHipGetSymbolSize = (hipGetSymbolSizeTy)__interception::LookupSymbol(
      Handle, "hipGetSymbolSize");
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
    if (pHipGetDeviceCount(&Count) == 0 && Count > 0) {
      DeviceArchNames = (char (*)[256])calloc(Count, sizeof(*DeviceArchNames));
      if (!DeviceArchNames) {
        PROF_ERR("%s\n", "failed to allocate device arch name table");
        return;
      }
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

#ifdef _WIN32
static BOOL CALLBACK ensureHipLoadedCb(PINIT_ONCE, PVOID, PVOID *) {
  doEnsureHipLoaded();
  return TRUE;
}
#endif

static void ensureHipLoaded(void) {
#ifdef _WIN32
  InitOnceExecuteOnce(&HipLoadedOnce, ensureHipLoadedCb, NULL, NULL);
#else
  pthread_once(&HipLoadedOnce, doEnsureHipLoaded);
#endif
}

/* -------------------------------------------------------------------------- */
/*  Public wrappers that forward to the loaded HIP symbols                   */
/* -------------------------------------------------------------------------- */

static int hipGetSymbolAddress(void **devPtr, const void *symbol) {
  ensureHipLoaded();
  return pHipGetSymbolAddress ? pHipGetSymbolAddress(devPtr, symbol) : -1;
}

static int hipGetSymbolSize(size_t *size, const void *symbol) {
  ensureHipLoaded();
  return pHipGetSymbolSize ? pHipGetSymbolSize(size, symbol) : -1;
}

static int hipMemcpy(void *dest, const void *src, size_t len,
                     int kind /*2=DToH*/) {
  ensureHipLoaded();
  return pHipMemcpy ? pHipMemcpy(dest, src, len, kind) : -1;
}

/* Device section symbols must be registered with CLR first; otherwise
 * hipMemcpy may take a CPU path and crash. */
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
  void *DeviceVar; /* device address of __llvm_profile_sections_<CUID> */
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
/*  ELF symbol enumeration (manual parse: compiler-rt cannot link LLVM Support)
 */
/* -------------------------------------------------------------------------- */

#if __has_include(<elf.h>)
#include <elf.h>

/* Callback invoked for every matching symbol name found in the ELF image.
 * Return 0 to continue iteration, non-zero to stop. */
typedef int (*SymbolCallback)(const char *Name, void *UserData);

/* If Image is a clang offload bundle, return a pointer to the first embedded
 * ELF. Returns Image if not a bundle, nullptr if a bundle holds no ELF. */
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
    Cursor += IDSize; /* skip entry ID */

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

/* Invoke CB for every global symbol in Image (an AMDGPU ELF or offload bundle)
 * whose name starts with PREFIX. Image may be null. */
static void enumerateElfSymbols(const void *Image, const char *Prefix,
                                SymbolCallback CB, void *UserData) {
  if (!Image)
    return;

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

/* Register one __llvm_profile_sections_<CUID> symbol on the module entry.
 * hipModuleGetGlobal also registers the device address with CLR so hipMemcpy
 * can copy from it later. */
static int registerPrfSymbol(const char *Name, void *UserData) {
  EnumState *S = (EnumState *)UserData;
  OffloadDynamicModuleInfo *MI = S->ModInfo;

  /* The symbol is the per-TU sections struct itself, not a pointer
   * indirection, so this address is the hipMemcpy source. */
  void *DeviceVar = nullptr;
  size_t Bytes = 0;
  if (hipModuleGetGlobal(&DeviceVar, &Bytes, S->Module, Name) != 0) {
    PROF_WARN("failed to get symbol %s for module %p\n", Name, S->Module);
    return 0; /* continue */
  }

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

  (void)Name;
  return 0; /* continue enumeration */
}

#endif /* __has_include(<elf.h>) */

/* -------------------------------------------------------------------------- */
/*  Registration / un-registration helpers                                   */
/* -------------------------------------------------------------------------- */

extern "C" void
__llvm_profile_offload_register_dynamic_module(int ModuleLoadRc, void **Ptr,
                                               const void *Image) {
  if (ModuleLoadRc)
    return;

  lockDynamicModules();

  if (isVerboseMode())
    PROF_NOTE("Registering loaded module %d: rc=%d, module=%p, image=%p\n",
              NumDynamicModules, ModuleLoadRc, *Ptr, Image);

  if (NumDynamicModules >= CapDynamicModules) {
    int NewCap = CapDynamicModules ? CapDynamicModules * 2 : 64;
    OffloadDynamicModuleInfo *New = (OffloadDynamicModuleInfo *)realloc(
        DynamicModules, NewCap * sizeof(OffloadDynamicModuleInfo));
    if (!New) {
      unlockDynamicModules();
      return;
    }
    DynamicModules = New;
    CapDynamicModules = NewCap;
  }

  OffloadDynamicModuleInfo *MI = &DynamicModules[NumDynamicModules++];
  MI->ModulePtr = *Ptr;
  MI->TUs = nullptr;
  MI->NumTUs = 0;
  MI->CapTUs = 0;

  /* Dynamic-module profiling needs ELF parsing for symbol enumeration. */
#if __has_include(<elf.h>)
  EnumState State = {*Ptr, MI};
  enumerateElfSymbols(Image, "__llvm_profile_sections_", registerPrfSymbol,
                      &State);
#else
  (void)Image;
  if (isVerboseMode())
    PROF_NOTE("%s",
              "Dynamic module profiling not supported on this platform\n");
#endif

  if (MI->NumTUs == 0) {
    PROF_WARN("no __llvm_profile_sections_* symbols found in module %p\n",
              *Ptr);
  } else if (isVerboseMode()) {
    PROF_NOTE("Module %p: registered %d TU(s)\n", *Ptr, MI->NumTUs);
  }

  unlockDynamicModules();
}

extern "C" void __llvm_profile_offload_unregister_dynamic_module(void *Ptr) {
  lockDynamicModules();
  for (int i = 0; i < NumDynamicModules; ++i) {
    OffloadDynamicModuleInfo *MI = &DynamicModules[i];

    /* HIP recycles hipModule_t addresses; drained slots are cleared so a
     * recycled handle finds the new slot, not the dead one. */
    if (MI->ModulePtr != Ptr)
      continue;

    if (isVerboseMode())
      PROF_NOTE("Unregistering module %p (%d TUs)\n", MI->ModulePtr,
                MI->NumTUs);

    static int NextTUIndex = 0;
    for (int t = 0; t < MI->NumTUs; ++t) {
      OffloadDynamicTUInfo *TU = &MI->TUs[t];
      if (TU->Processed) {
        if (isVerboseMode())
          PROF_NOTE("Module %p TU %d already processed, skipping\n", Ptr, t);
        continue;
      }
      int TUIndex = __atomic_fetch_add(&NextTUIndex, 1, __ATOMIC_RELAXED);
      if (TU->DeviceVar) {
        int CurDev = 0;
        hipGetDevice(&CurDev);
        const char *ArchName = getDeviceArchName(CurDev);
        /* Encode TUIndex in Target so each drain writes a distinct profraw;
         * otherwise back-to-back drains overwrite the same file. */
        char TargetWithTU[64];
        snprintf(TargetWithTU, sizeof(TargetWithTU), "%s.%d", ArchName,
                 TUIndex);
        if (processDeviceOffloadPrf(TU->DeviceVar, TargetWithTU, nullptr) == 0)
          TU->Processed = 1;
        else
          PROF_WARN("failed to process profile data for module %p TU %d\n", Ptr,
                    t);
      }
    }
    MI->ModulePtr = nullptr;
    unlockDynamicModules();
    return;
  }

  if (isVerboseMode())
    PROF_WARN("unregister called for unknown module %p\n", Ptr);
  unlockDynamicModules();
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

struct OffloadSectionShadow {
  void *Data;
  void *Counters;
  void *Names;
};

struct OffloadSectionShadowGroup {
  OffloadSectionShadow *Shadows;
  int NumShadows;
  int CapShadows;
  int NumSections;
};

static OffloadSectionShadowGroup *OffloadSectionShadowGroups = nullptr;
static int CapSectionShadowGroups = 0;

static int ensureSectionShadowGroupCapacity(void) {
  if (CapSectionShadowGroups >= CapShadowVariables)
    return 0;
  OffloadSectionShadowGroup *New = (OffloadSectionShadowGroup *)realloc(
      OffloadSectionShadowGroups, CapShadowVariables * sizeof(*New));
  if (!New)
    return -1;
  __builtin_memset(New + CapSectionShadowGroups, 0,
                   (CapShadowVariables - CapSectionShadowGroups) *
                       sizeof(*New));
  OffloadSectionShadowGroups = New;
  CapSectionShadowGroups = CapShadowVariables;
  return 0;
}

static int ensureSectionShadowCapacity(OffloadSectionShadowGroup *Group,
                                       int MinCapacity) {
  if (Group->CapShadows >= MinCapacity)
    return 0;
  int NewCap = Group->CapShadows ? Group->CapShadows * 2 : 4;
  while (NewCap < MinCapacity)
    NewCap *= 2;
  OffloadSectionShadow *New =
      (OffloadSectionShadow *)realloc(Group->Shadows, NewCap * sizeof(*New));
  if (!New)
    return -1;
  __builtin_memset(New + Group->CapShadows, 0,
                   (NewCap - Group->CapShadows) * sizeof(*New));
  Group->Shadows = New;
  Group->CapShadows = NewCap;
  return 0;
}

extern "C" void __llvm_profile_offload_register_shadow_variable(void *ptr) {
  if (growPtrArray(&OffloadShadowVariables, &NumShadowVariables,
                   &CapShadowVariables, 64))
    return;
  if (ensureSectionShadowGroupCapacity())
    return;
  int Index = NumShadowVariables++;
  OffloadShadowVariables[Index] = ptr;
  __builtin_memset(&OffloadSectionShadowGroups[Index], 0,
                   sizeof(OffloadSectionShadowGroups[Index]));
}

extern "C" void
__llvm_profile_offload_register_section_shadow_variable(void *ptr) {
  if (NumShadowVariables == 0)
    return;

  OffloadSectionShadowGroup *Group =
      &OffloadSectionShadowGroups[NumShadowVariables - 1];
  int ShadowIndex = Group->NumSections / 3;
  if (ensureSectionShadowCapacity(Group, ShadowIndex + 1))
    return;
  if (ShadowIndex >= Group->NumShadows)
    Group->NumShadows = ShadowIndex + 1;

  OffloadSectionShadow *Shadow = &Group->Shadows[ShadowIndex];
  switch (Group->NumSections % 3) {
  case 0:
    Shadow->Data = ptr;
    break;
  case 1:
    Shadow->Counters = ptr;
    break;
  case 2:
    Shadow->Names = ptr;
    break;
  }
  ++Group->NumSections;
}

namespace {

// free()-based scope guard. Use .release() to transfer ownership.
struct UniqueFree {
  void *Ptr;
  explicit UniqueFree(void *P = nullptr) : Ptr(P) {}
  ~UniqueFree() { free(Ptr); }
  UniqueFree(const UniqueFree &) = delete;
  UniqueFree &operator=(const UniqueFree &) = delete;
  char *get() const { return static_cast<char *>(Ptr); }
  void reset(void *P) {
    free(Ptr);
    Ptr = P;
  }
  void *release() {
    void *P = Ptr;
    Ptr = nullptr;
    return P;
  }
};

} // namespace

static int getRegisteredSectionBounds(void *Shadow, void **DevicePtr,
                                      size_t *Size) {
  *DevicePtr = nullptr;
  *Size = 0;
  int AddrRc = hipGetSymbolAddress(DevicePtr, Shadow);
  int SizeRc = hipGetSymbolSize(Size, Shadow);
  return AddrRc == 0 && SizeRc == 0 && *DevicePtr && *Size > 0 ? 0 : -1;
}

struct RegisteredSectionRange {
  const void *Data;
  const void *Counters;
  const void *Names;
  size_t DataSize;
  size_t CountersSize;
  size_t NamesSize;
  size_t DataOffset;
  size_t CountersOffset;
  size_t NamesOffset;
};

static int
hasCompleteSectionShadows(const OffloadSectionShadowGroup *Sections) {
  if (!Sections || Sections->NumShadows == 0 || Sections->NumSections % 3 != 0)
    return 0;
  for (int I = 0; I < Sections->NumShadows; ++I) {
    if (!Sections->Shadows[I].Data || !Sections->Shadows[I].Counters ||
        !Sections->Shadows[I].Names)
      return 0;
  }
  return 1;
}

static int processDeviceOffloadPrf(void *DeviceOffloadPrf, const char *Target,
                                   const OffloadSectionShadowGroup *Sections) {
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

  int UseRegisteredSections = hasCompleteSectionShadows(Sections);
  RegisteredSectionRange *RegisteredRanges = nullptr;
  int NumRegisteredRanges = 0;

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

  // Owns freshly malloc'd buffers; release() transfers ownership to the cache.
  UniqueFree CntsOwner, DataOwner, NamesOwner, RegisteredRangeOwner;

  if (UseRegisteredSections) {
    NumRegisteredRanges = Sections->NumShadows;
    RegisteredRangeOwner.reset(
        malloc(NumRegisteredRanges * sizeof(RegisteredSectionRange)));
    RegisteredRanges = (RegisteredSectionRange *)RegisteredRangeOwner.get();
    if (!RegisteredRanges) {
      PROF_ERR("%s\n", "failed to allocate registered section table");
      return -1;
    }
    __builtin_memset(RegisteredRanges, 0,
                     NumRegisteredRanges * sizeof(*RegisteredRanges));

    size_t RegisteredDataSize = 0;
    size_t RegisteredCountersSize = 0;
    size_t RegisteredNamesSize = 0;
    for (int I = 0; I < NumRegisteredRanges; ++I) {
      void *Data = nullptr;
      void *Counters = nullptr;
      void *Names = nullptr;
      size_t ThisDataSize = 0;
      size_t ThisCountersSize = 0;
      size_t ThisNamesSize = 0;
      OffloadSectionShadow *Shadow = &Sections->Shadows[I];
      if (getRegisteredSectionBounds(Shadow->Data, &Data, &ThisDataSize) != 0 ||
          getRegisteredSectionBounds(Shadow->Counters, &Counters,
                                     &ThisCountersSize) != 0 ||
          getRegisteredSectionBounds(Shadow->Names, &Names, &ThisNamesSize) !=
              0) {
        PROF_ERR("%s\n", "failed to get registered section bounds");
        return -1;
      }

      RegisteredRanges[I].Data = Data;
      RegisteredRanges[I].Counters = Counters;
      RegisteredRanges[I].Names = Names;
      RegisteredRanges[I].DataSize = ThisDataSize;
      RegisteredRanges[I].CountersSize = ThisCountersSize;
      RegisteredRanges[I].NamesSize = ThisNamesSize;
      RegisteredRanges[I].DataOffset = RegisteredDataSize;
      RegisteredRanges[I].CountersOffset = RegisteredCountersSize;
      RegisteredDataSize += ThisDataSize;
      RegisteredCountersSize += ThisCountersSize;

      int ReuseNames = 0;
      for (int J = 0; J < I; ++J) {
        if (RegisteredRanges[J].Names == Names &&
            RegisteredRanges[J].NamesSize == ThisNamesSize) {
          RegisteredRanges[I].NamesOffset = RegisteredRanges[J].NamesOffset;
          ReuseNames = 1;
          break;
        }
      }
      if (!ReuseNames) {
        RegisteredRanges[I].NamesOffset = RegisteredNamesSize;
        RegisteredNamesSize += ThisNamesSize;
      }
    }

    DataSize = RegisteredDataSize;
    CountersSize = RegisteredCountersSize;
    NamesSize = RegisteredNamesSize;
    HostDataBegin = (char *)malloc(DataSize);
    HostCountersBegin = (char *)malloc(CountersSize);
    HostNamesBegin = NamesSize ? (char *)malloc(NamesSize) : nullptr;
    DataOwner.reset(HostDataBegin);
    CntsOwner.reset(HostCountersBegin);
    NamesOwner.reset(HostNamesBegin);
    if ((DataSize > 0 && !HostDataBegin) ||
        (CountersSize > 0 && !HostCountersBegin) ||
        (NamesSize > 0 && !HostNamesBegin)) {
      PROF_ERR("%s\n", "failed to allocate host memory for device sections");
      return -1;
    }

    for (int I = 0; I < NumRegisteredRanges; ++I) {
      RegisteredSectionRange *R = &RegisteredRanges[I];
      if (memcpyDeviceToHost(HostDataBegin + R->DataOffset, R->Data,
                             R->DataSize) != 0 ||
          memcpyDeviceToHost(HostCountersBegin + R->CountersOffset, R->Counters,
                             R->CountersSize) != 0) {
        PROF_ERR("%s\n", "failed to copy profile sections from device");
        return -1;
      }

      int CopyNames = 1;
      for (int J = 0; J < I; ++J) {
        if (RegisteredRanges[J].Names == R->Names &&
            RegisteredRanges[J].NamesSize == R->NamesSize) {
          CopyNames = 0;
          break;
        }
      }
      if (CopyNames && R->NamesSize > 0 &&
          memcpyDeviceToHost(HostNamesBegin + R->NamesOffset, R->Names,
                             R->NamesSize) != 0) {
        PROF_ERR("%s\n", "failed to copy profile sections from device");
        return -1;
      }
    }
  } else {
    if (CountersSize > 0 && DevCntsBegin == CachedDevCntsBegin &&
        CountersSize == CachedCntsSize) {
      HostCountersBegin = CachedHostCnts;
      CntsReused = 1;
      if (isVerboseMode())
        PROF_NOTE("Reusing cached counters section (%zu bytes)\n",
                  CountersSize);
    } else if (CountersSize > 0) {
      HostCountersBegin = (char *)malloc(CountersSize);
      CntsOwner.reset(HostCountersBegin);
    }

    if (DataSize > 0 && DevDataBegin == CachedDevDataBegin &&
        DataSize == CachedDataSize) {
      HostDataBegin = CachedHostData;
      DataReused = 1;
      if (isVerboseMode())
        PROF_NOTE("Reusing cached data section (%zu bytes)\n", DataSize);
    } else if (DataSize > 0) {
      HostDataBegin = (char *)malloc(DataSize);
      DataOwner.reset(HostDataBegin);
    }

    if (NamesSize > 0 && DevNamesBegin == CachedDevNamesBegin &&
        NamesSize == CachedNamesSize) {
      HostNamesBegin = CachedHostNames;
      NamesReused = 1;
      if (isVerboseMode())
        PROF_NOTE("Reusing cached names section (%zu bytes)\n", NamesSize);
    } else if (NamesSize > 0) {
      HostNamesBegin = (char *)malloc(NamesSize);
      NamesOwner.reset(HostNamesBegin);
    }

    if ((DataSize > 0 && !HostDataBegin) ||
        (CountersSize > 0 && !HostCountersBegin) ||
        (NamesSize > 0 && !HostNamesBegin)) {
      PROF_ERR("%s\n", "failed to allocate host memory for device sections");
      return -1;
    }

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

    /* Cache buffers so RDC-mode multi-shadow drains can reuse them.
     * release() prevents the scope guards from freeing what the cache owns. */
    if (!CntsReused && CountersSize > 0) {
      CachedDevCntsBegin = DevCntsBegin;
      CachedHostCnts = HostCountersBegin;
      CachedCntsSize = CountersSize;
      CntsOwner.release();
    }
    if (!DataReused && DataSize > 0) {
      CachedDevDataBegin = DevDataBegin;
      CachedHostData = HostDataBegin;
      CachedDataSize = DataSize;
      DataOwner.release();
    }
    if (!NamesReused && NamesSize > 0) {
      CachedDevNamesBegin = DevNamesBegin;
      CachedHostNames = HostNamesBegin;
      CachedNamesSize = NamesSize;
      NamesOwner.release();
    }
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
  UniqueFree ContiguousBuf(malloc(ContiguousBufferSize));
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

  // CounterPtr is a device-relative offset; relocate it for the file layout
  // where the Data section precedes Counters.
  __llvm_profile_data *RelocatedData = (__llvm_profile_data *)BufDataBegin;
  for (uint64_t i = 0; i < NumData; ++i) {
    if (RelocatedData[i].CounterPtr) {
      ptrdiff_t DeviceCounterPtrOffset = (ptrdiff_t)RelocatedData[i].CounterPtr;
      size_t DataRecordOffset = i * sizeof(__llvm_profile_data);
      const char *RangeDevDataBegin = (const char *)DevDataBegin;
      const char *RangeDevCountersBegin = (const char *)DevCntsBegin;
      size_t RangeCountersOffset = 0;
      if (UseRegisteredSections) {
        int FoundRange = 0;
        for (int R = 0; R < NumRegisteredRanges; ++R) {
          RegisteredSectionRange *Range = &RegisteredRanges[R];
          if (DataRecordOffset < Range->DataOffset ||
              DataRecordOffset >= Range->DataOffset + Range->DataSize)
            continue;
          RangeDevDataBegin = (const char *)Range->Data;
          RangeDevCountersBegin = (const char *)Range->Counters;
          RangeCountersOffset = Range->CountersOffset;
          DataRecordOffset -= Range->DataOffset;
          FoundRange = 1;
          break;
        }
        if (!FoundRange) {
          PROF_ERR("%s\n", "failed to locate profile data record range");
          return -1;
        }
      }
      const char *DeviceDataStructAddr = RangeDevDataBegin + DataRecordOffset;
      const char *DeviceCountersAddr =
          DeviceDataStructAddr + DeviceCounterPtrOffset;
      ptrdiff_t OffsetIntoCountersSection =
          DeviceCountersAddr - RangeDevCountersBegin;

      ptrdiff_t NewRelativeOffset =
          DataSize + PaddingBytesBeforeCounters + RangeCountersOffset +
          OffsetIntoCountersSection - (i * sizeof(__llvm_profile_data));
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

static int processShadowVariable(int Index, const char *Target) {
  void *ShadowVar = OffloadShadowVariables[Index];
  void *DeviceSections = nullptr;
  if (hipGetSymbolAddress(&DeviceSections, ShadowVar) != 0) {
    PROF_WARN("failed to get symbol address for shadow variable %p\n",
              ShadowVar);
    return -1;
  }
  /* DeviceSections points at the per-TU sections struct itself. */
  const OffloadSectionShadowGroup *Sections = nullptr;
  if (Index < CapSectionShadowGroups)
    Sections = &OffloadSectionShadowGroups[Index];
  if (!hasCompleteSectionShadows(Sections))
    return 0;
  return processDeviceOffloadPrf(DeviceSections, Target, Sections);
}

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

  /* Shadow variables (static-linked kernels): drain from every device. */
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
        /* RDC-mode multi-shadow drains need a distinct profraw per TU;
         * single-TU programs keep the bare arch target. */
        const char *Target = ArchName;
        char TargetWithIdx[64];
        if (NumShadowVariables > 1) {
          snprintf(TargetWithIdx, sizeof(TargetWithIdx), "%s.%d", ArchName, i);
          Target = TargetWithIdx;
        }
        if (processShadowVariable(i, Target) != 0)
          Ret = -1;
      }
    }

    if (OrigDevice >= 0)
      hipSetDevice(OrigDevice);
  }

  /* Warn about unprocessed TUs; skip cleared slots (already drained). */
  lockDynamicModules();
  for (int i = 0; i < NumDynamicModules; ++i) {
    OffloadDynamicModuleInfo *MI = &DynamicModules[i];
    if (!MI->ModulePtr)
      continue;
    for (int t = 0; t < MI->NumTUs; ++t) {
      if (!MI->TUs[t].Processed) {
        PROF_WARN("dynamic module %p TU %d was not processed before exit\n",
                  MI->ModulePtr, t);
        Ret = -1;
      }
    }
  }
  unlockDynamicModules();

  if (Ret != 0)
    PROF_WARN("%s\n", "failed to collect device profile data");
  return Ret;
}

/* Interceptors for hipModuleLoad* / hipModuleUnload. Linux only. */

#if defined(__linux__) && !defined(_WIN32)

INTERCEPTOR(int, hipModuleLoad, void **module, const char *fname) {
  int rc = REAL(hipModuleLoad)(module, fname);
  /* Pass NULL image: no in-memory ELF is available for filename loads,
   * so the register hook skips symbol enumeration. */
  __llvm_profile_offload_register_dynamic_module(rc, module, nullptr);
  return rc;
}

INTERCEPTOR(int, hipModuleLoadData, void **module, const void *image) {
  int rc = REAL(hipModuleLoadData)(module, image);
  __llvm_profile_offload_register_dynamic_module(rc, module, image);
  return rc;
}

INTERCEPTOR(int, hipModuleLoadDataEx, void **module, const void *image,
            unsigned numOptions, void **options, void **optionValues) {
  int rc = REAL(hipModuleLoadDataEx)(module, image, numOptions, options,
                                     optionValues);
  __llvm_profile_offload_register_dynamic_module(rc, module, image);
  return rc;
}

INTERCEPTOR(int, hipModuleUnload, void *module) {
  /* Drain counters before the module is destroyed; device addresses
   * captured at register time are invalid after unload. */
  __llvm_profile_offload_unregister_dynamic_module(module);
  return REAL(hipModuleUnload)(module);
}

__attribute__((constructor)) static void installHipModuleInterceptors() {
  /* Skip when the HIP runtime is not loaded. INTERCEPT_FUNCTION uses the
   * sanitizer interception framework, which can perturb dlsym/PLT state for
   * the rest of the process even when the target symbol is absent; non-HIP
   * programs linked with libclang_rt.profile.a must see zero side effects. */
  if (!dlsym(RTLD_DEFAULT, "hipModuleLoad"))
    return;
  if (!INTERCEPT_FUNCTION(hipModuleLoad))
    return;
  if (isVerboseMode())
    PROF_NOTE("%s", "Installing hipModuleLoad*/hipModuleUnload interceptors\n");
  INTERCEPT_FUNCTION(hipModuleLoadData);
  INTERCEPT_FUNCTION(hipModuleLoadDataEx);
  INTERCEPT_FUNCTION(hipModuleUnload);
}

#endif /* __linux__ */
