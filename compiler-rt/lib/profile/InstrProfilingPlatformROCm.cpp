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
#include <wchar.h>
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
// windows.h needs to be included before tlhelp32.h.
#include <tlhelp32.h>
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

#if defined(__linux__) && !defined(_WIN32)
// Record a drained section-bounds tuple so the supplemental HSA-introspection
// pass (Linux only) skips any code object the host-shadow path already
// drained. Defined alongside the HSA drain below; forward-declared here so
// processDeviceOffloadPrf can register every successful host-shadow drain.
static void profRecordDrainedBounds(const void *Data, const void *Counters,
                                    const void *Names);
#endif

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
#if defined(__linux__) && !defined(_WIN32)
typedef void *HipStream;
typedef int (*hipStreamGetDeviceTy)(HipStream, int *);
#endif

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
#if defined(__linux__) && !defined(_WIN32)
static hipStreamGetDeviceTy pHipStreamGetDevice = nullptr;
#endif
static hipGetDevicePropertiesTy pHipGetDeviceProperties = nullptr;

static int NumDevices = 0;
/* 256 matches hipDeviceProp_t::gcnArchName, the source field width. */
static char (*DeviceArchNames)[256] = nullptr;
#if defined(__linux__) && !defined(_WIN32)
static unsigned char *UsedDevices = nullptr;
static int AnyDeviceUsed = 0;
#endif

#ifdef _WIN32
static wchar_t toLowerAsciiW(wchar_t C) {
  return C >= L'A' && C <= L'Z' ? C - L'A' + L'a' : C;
}

static int wcsEqualNoCase(const wchar_t *A, const wchar_t *B) {
  while (*A && *B) {
    if (toLowerAsciiW(*A) != toLowerAsciiW(*B))
      return 0;
    ++A;
    ++B;
  }
  return *A == *B;
}

static int wcsStartsWithNoCase(const wchar_t *S, const wchar_t *Prefix) {
  while (*Prefix) {
    if (toLowerAsciiW(*S) != toLowerAsciiW(*Prefix))
      return 0;
    ++S;
    ++Prefix;
  }
  return 1;
}

static int wcsEndsWithNoCase(const wchar_t *S, const wchar_t *Suffix) {
  size_t SLen = wcslen(S);
  size_t SuffixLen = wcslen(Suffix);
  return SLen >= SuffixLen && wcsEqualNoCase(S + SLen - SuffixLen, Suffix);
}

static int isHipRuntimeModuleName(const wchar_t *Name) {
  return wcsEqualNoCase(Name, L"amdhip64.dll") ||
         (wcsStartsWithNoCase(Name, L"amdhip64_") &&
          wcsEndsWithNoCase(Name, L".dll"));
}

static void *findLoadedHipRuntime(void) {
  HMODULE Handle = GetModuleHandleW(L"amdhip64.dll");
  if (Handle)
    return (void *)Handle;

  HANDLE Snapshot = CreateToolhelp32Snapshot(
      TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId());
  if (Snapshot == INVALID_HANDLE_VALUE)
    return nullptr;

  MODULEENTRY32W Entry;
  Entry.dwSize = sizeof(Entry);
  if (Module32FirstW(Snapshot, &Entry)) {
    do {
      if (isHipRuntimeModuleName(Entry.szModule)) {
        Handle = Entry.hModule;
        break;
      }
    } while (Module32NextW(Snapshot, &Entry));
  }

  CloseHandle(Snapshot);
  return (void *)Handle;
}
#endif

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
  /* Use the app's loaded HIP runtime to avoid binding another ROCm version. */
  void *Handle = findLoadedHipRuntime();
#else
  const char *HipLibName = "libamdhip64.so";
  void *Handle = __interception::OpenLibrary(HipLibName);
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
#if defined(__linux__) && !defined(_WIN32)
  pHipStreamGetDevice = (hipStreamGetDeviceTy)__interception::LookupSymbol(
      Handle, "hipStreamGetDevice");
#endif
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
#if defined(__linux__) && !defined(_WIN32)
      UsedDevices = (unsigned char *)calloc(Count, sizeof(*UsedDevices));
      if (!UsedDevices && isVerboseMode())
        PROF_NOTE("%s\n", "Device-use tracking disabled");
#endif
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

#if defined(__linux__) && !defined(_WIN32)
static int hipStreamGetDevice(HipStream Stream, int *DeviceId) {
  ensureHipLoaded();
  return pHipStreamGetDevice ? pHipStreamGetDevice(Stream, DeviceId) : -1;
}

static void markDeviceUsed(int DeviceId) {
  if (DeviceId < 0 || DeviceId >= NumDevices || !UsedDevices)
    return;
  __atomic_store_n(&UsedDevices[DeviceId], 1, __ATOMIC_RELAXED);
  __atomic_store_n(&AnyDeviceUsed, 1, __ATOMIC_RELEASE);
}

static void markCurrentDeviceUsed(void) {
  int DeviceId = -1;
  if (hipGetDevice(&DeviceId) == 0)
    markDeviceUsed(DeviceId);
}

static void markLaunchStreamDeviceUsed(HipStream Stream) {
  int DeviceId = -1;
  if (Stream && hipStreamGetDevice(Stream, &DeviceId) == 0) {
    markDeviceUsed(DeviceId);
    return;
  }
  markCurrentDeviceUsed();
}

static int shouldCollectDevice(int DeviceId) {
  if (UsedDevices && __atomic_load_n(&AnyDeviceUsed, __ATOMIC_ACQUIRE) &&
      !__atomic_load_n(&UsedDevices[DeviceId], __ATOMIC_RELAXED))
    return 0;
  return 1;
}
#else
static int shouldCollectDevice(int) { return 1; }
#endif

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

  /* Match CGCUDANV.cpp: data, counters, then names for each kernel. */
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
  } else {
#if defined(__linux__) && !defined(_WIN32)
    // Dedup against the supplemental HSA pass: this section is now drained, so
    // the HSA walk must not drain the same device code object again.
    profRecordDrainedBounds(DevDataBegin, DevCntsBegin, DevNamesBegin);
#endif
    if (isVerboseMode())
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

/* ========================================================================== */
/*  Supplemental HSA-introspection drain (Linux only)                         */
/*                                                                            */
/*  The host-shadow drain above only sees device code objects registered      */
/*  host-side (__hipRegisterVar shadows) or loaded through an intercepted */
/*  hipModuleLoad* call. Device code linked by the offload device linker with */
/*  no host-side shadow -- e.g. RCCL, whose many device functions are glued */
/*  into a single kernel with no source module -- is invisible to it. This */
/*  pass walks every GPU agent's loaded executables via HSA, finds each */
/*  __llvm_profile_sections table directly on the device, and drains the ones */
/*  the host-shadow pass did not already handle (deduped by the device */
/*  section-bounds tuple). It reuses processDeviceOffloadPrf() for the */
/*  copy/relocate/write so the on-disk profraw layout is identical.           */
/* ========================================================================== */
#if defined(__linux__) && !defined(_WIN32)

/* Minimal HSA type/enum stubs. compiler-rt cannot depend on ROCm headers at
 * build time, so mirror just the handful of HSA declarations the drain needs.
 * Values match hsa/hsa.h and hsa/hsa_ven_amd_loader.h. */
typedef uint32_t prof_hsa_status_t;
#define PROF_HSA_STATUS_SUCCESS ((prof_hsa_status_t)0x0)
#define PROF_HSA_STATUS_INFO_BREAK ((prof_hsa_status_t)0x1)

typedef struct {
  uint64_t handle;
} prof_hsa_agent_t;
typedef struct {
  uint64_t handle;
} prof_hsa_executable_t;
typedef struct {
  uint64_t handle;
} prof_hsa_executable_symbol_t;

typedef uint32_t prof_hsa_agent_info_t;
#define PROF_HSA_AGENT_INFO_NAME ((prof_hsa_agent_info_t)0)
#define PROF_HSA_AGENT_INFO_DEVICE ((prof_hsa_agent_info_t)17)

typedef uint32_t prof_hsa_device_type_t;
#define PROF_HSA_DEVICE_TYPE_GPU ((prof_hsa_device_type_t)1)

typedef uint32_t prof_hsa_symbol_kind_t;
#define PROF_HSA_SYMBOL_KIND_VARIABLE ((prof_hsa_symbol_kind_t)0)

typedef uint32_t prof_hsa_executable_symbol_info_t;
#define PROF_HSA_EXECUTABLE_SYMBOL_INFO_TYPE                                   \
  ((prof_hsa_executable_symbol_info_t)0)
#define PROF_HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH                            \
  ((prof_hsa_executable_symbol_info_t)1)
#define PROF_HSA_EXECUTABLE_SYMBOL_INFO_NAME                                   \
  ((prof_hsa_executable_symbol_info_t)2)
#define PROF_HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS                       \
  ((prof_hsa_executable_symbol_info_t)21)

#define PROF_HSA_EXTENSION_AMD_LOADER ((uint16_t)0x201)

typedef uint32_t prof_hsa_loader_storage_type_t;

typedef struct {
  prof_hsa_agent_t agent;
  prof_hsa_executable_t executable;
  prof_hsa_loader_storage_type_t code_object_storage_type;
  const void *code_object_storage_base;
  size_t code_object_storage_size;
  size_t code_object_storage_offset;
  const void *segment_base;
  size_t segment_size;
} prof_hsa_loader_segment_descriptor_t;

typedef prof_hsa_status_t (*hsa_init_ty)(void);
typedef prof_hsa_status_t (*hsa_iterate_agents_ty)(
    prof_hsa_status_t (*)(prof_hsa_agent_t, void *), void *);
typedef prof_hsa_status_t (*hsa_agent_get_info_ty)(prof_hsa_agent_t,
                                                   prof_hsa_agent_info_t,
                                                   void *);
typedef prof_hsa_status_t (*hsa_executable_iterate_agent_symbols_ty)(
    prof_hsa_executable_t, prof_hsa_agent_t,
    prof_hsa_status_t (*)(prof_hsa_executable_t, prof_hsa_agent_t,
                          prof_hsa_executable_symbol_t, void *),
    void *);
typedef prof_hsa_status_t (*hsa_executable_symbol_get_info_ty)(
    prof_hsa_executable_symbol_t, prof_hsa_executable_symbol_info_t, void *);
typedef prof_hsa_status_t (*hsa_system_get_major_extension_table_ty)(uint16_t,
                                                                     uint16_t,
                                                                     size_t,
                                                                     void *);
typedef prof_hsa_status_t (*hsa_loader_query_segment_descriptors_ty)(
    prof_hsa_loader_segment_descriptor_t *, size_t *);

/* First two members of hsa_ven_amd_loader_1_00_pfn_t. Only
 * query_segment_descriptors is used; query_host_address keeps the offset. */
typedef struct {
  void *query_host_address;
  hsa_loader_query_segment_descriptors_ty query_segment_descriptors;
} prof_hsa_loader_pfn_t;

static hsa_iterate_agents_ty pHsaIterateAgents = nullptr;
static hsa_agent_get_info_ty pHsaAgentGetInfo = nullptr;
static hsa_executable_iterate_agent_symbols_ty pHsaExecIterAgentSyms = nullptr;
static hsa_executable_symbol_get_info_ty pHsaSymGetInfo = nullptr;
static hsa_loader_query_segment_descriptors_ty pQuerySegDescs = nullptr;

/* 0 = not yet attempted, 1 = ready, -1 = unavailable. Accessed with acquire/
 * release atomics: a thread observing HsaRuntimeState==1 (acquire) also sees
 * the fully-written p* function pointers (published before the release store
 * of HsaRuntimeState=1 below). */
static int HsaRuntimeState = 0;

static int setHsaRuntimeState(int S) {
  __atomic_store_n(&HsaRuntimeState, S, __ATOMIC_RELEASE);
  return S > 0 ? 0 : -1;
}

/* Resolve HSA entry points (and the AMD loader extension) once, and confirm
 * HIP's hipMemcpy is reachable for the device-to-host copies. HIP itself is
 * resolved by the shared ensureHipLoaded() above. */
static int loadHsaRuntimePointers(void) {
  int State = __atomic_load_n(&HsaRuntimeState, __ATOMIC_ACQUIRE);
  if (State)
    return State > 0 ? 0 : -1;

  if (!__interception::DynamicLoaderAvailable()) {
    if (isVerboseMode())
      PROF_NOTE("%s", "Dynamic library loading not available - "
                      "HSA device profiling disabled\n");
    return setHsaRuntimeState(-1);
  }

  void *Hsa = __interception::OpenLibrary("libhsa-runtime64.so");
  if (!Hsa)
    Hsa = __interception::OpenLibrary("libhsa-runtime64.so.1");
  if (!Hsa) {
    if (isVerboseMode())
      PROF_NOTE("%s", "libhsa-runtime64.so not loadable - "
                      "HSA device profiling disabled\n");
    return setHsaRuntimeState(-1);
  }

  hsa_init_ty pHsaInit =
      (hsa_init_ty)__interception::LookupSymbol(Hsa, "hsa_init");
  hsa_system_get_major_extension_table_ty pGetExtTable =
      (hsa_system_get_major_extension_table_ty)__interception::LookupSymbol(
          Hsa, "hsa_system_get_major_extension_table");
  pHsaIterateAgents = (hsa_iterate_agents_ty)__interception::LookupSymbol(
      Hsa, "hsa_iterate_agents");
  pHsaAgentGetInfo = (hsa_agent_get_info_ty)__interception::LookupSymbol(
      Hsa, "hsa_agent_get_info");
  pHsaExecIterAgentSyms =
      (hsa_executable_iterate_agent_symbols_ty)__interception::LookupSymbol(
          Hsa, "hsa_executable_iterate_agent_symbols");
  pHsaSymGetInfo =
      (hsa_executable_symbol_get_info_ty)__interception::LookupSymbol(
          Hsa, "hsa_executable_symbol_get_info");

  if (!pHsaInit || !pGetExtTable || !pHsaIterateAgents || !pHsaAgentGetInfo ||
      !pHsaExecIterAgentSyms || !pHsaSymGetInfo) {
    PROF_WARN("%s",
              "required HSA symbols missing - HSA device profiling disabled\n");
    return setHsaRuntimeState(-1);
  }

  /* Bring HSA up (idempotent, refcounted). This runs lazily on the first drain
   * rather than from the library constructor, so merely loading the
   * instrumented library does not initialize HSA in the process -- which would
   * break fork-based callers that deliberately keep HIP/HSA uninitialized in
   * the parent (see the constructor note at the end of the HSA block). In the
   * common case the drain runs from the profile write path while HSA is still
   * alive; if it only runs after HSA's own atexit(hsa_shut_down) has executed,
   * this simply re-initializes HSA (the process is exiting anyway). */
  prof_hsa_status_t St = pHsaInit();
  if (St != PROF_HSA_STATUS_SUCCESS && St != PROF_HSA_STATUS_INFO_BREAK) {
    if (isVerboseMode())
      PROF_NOTE("hsa_init failed (0x%x) - HSA device profiling disabled\n", St);
    return setHsaRuntimeState(-1);
  }

  prof_hsa_loader_pfn_t LoaderApi;
  __builtin_memset(&LoaderApi, 0, sizeof(LoaderApi));
  St = pGetExtTable(PROF_HSA_EXTENSION_AMD_LOADER, 1, sizeof(LoaderApi),
                    &LoaderApi);
  if (St != PROF_HSA_STATUS_SUCCESS || !LoaderApi.query_segment_descriptors) {
    PROF_WARN("AMD loader extension unavailable (0x%x) - "
              "HSA device profiling disabled\n",
              St);
    return setHsaRuntimeState(-1);
  }
  pQuerySegDescs = LoaderApi.query_segment_descriptors;

  /* The device-to-host copies go through the shared HIP loader. */
  ensureHipLoaded();
  if (!pHipMemcpy) {
    PROF_WARN("%s", "hipMemcpy unavailable - HSA device profiling disabled\n");
    return setHsaRuntimeState(-1);
  }

  if (isVerboseMode())
    PROF_NOTE("%s", "HSA + HIP runtime resolved for device profiling\n");
  return setHsaRuntimeState(1);
}

/* The canonical device bounds-table symbol from InstrProfilingPlatformGPU.c. */
static const char ProfileSectionsSymbol[] = "__llvm_profile_sections";

/* Dedup of drained section-bounds tuples, shared with the host-shadow path
 * (processDeviceOffloadPrf records here on every successful drain). A single
 * linked device code object exposes one __llvm_profile_sections, but the same
 * bounds may be seen via multiple agents, so each unique counter set is
 * drained exactly once across both paths. */
namespace {
struct ProfBoundsTuple {
  const void *data;
  const void *cnts;
  const void *names;
};
} // namespace

#define PROF_MAX_SEEN_BOUNDS 256
static ProfBoundsTuple SeenBounds[PROF_MAX_SEEN_BOUNDS];
static int NumSeenBounds = 0;

/* Pure check: has this bounds tuple already been drained? Does not mutate
 * state, so a transient failure does not permanently suppress retries. */
static int profBoundsAlreadyDrained(const void *D, const void *C,
                                    const void *N) {
  for (int i = 0; i < NumSeenBounds; ++i)
    if (SeenBounds[i].data == D && SeenBounds[i].cnts == C &&
        SeenBounds[i].names == N)
      return 1;
  return 0;
}

/* Record a drained bounds tuple. Idempotent. Called after a successful drain
 * (either path) so a failed attempt stays retryable. */
static void profRecordDrainedBounds(const void *D, const void *C,
                                    const void *N) {
  if (profBoundsAlreadyDrained(D, C, N))
    return;
  if (NumSeenBounds < PROF_MAX_SEEN_BOUNDS) {
    SeenBounds[NumSeenBounds].data = D;
    SeenBounds[NumSeenBounds].cnts = C;
    SeenBounds[NumSeenBounds].names = N;
    NumSeenBounds++;
  }
}

#define PROF_MAX_GPU_AGENTS 64

namespace {
struct GpuAgent {
  prof_hsa_agent_t agent;
  char arch[64];
};

struct WalkState {
  GpuAgent agents[PROF_MAX_GPU_AGENTS];
  int num_agents;
  int total_found;
  int total_drained;
};

/* Per (agent, executable) symbol-iteration state. */
struct SymbolState {
  const char *arch;
  int found;
  int drained;
};
} // namespace

/* HSA per-symbol callback: when it finds a __llvm_profile_sections variable,
 * drain it via processDeviceOffloadPrf() unless the host-shadow path (or an
 * earlier agent) already handled the same bounds. */
static prof_hsa_status_t onSymbol(prof_hsa_executable_t, prof_hsa_agent_t,
                                  prof_hsa_executable_symbol_t Sym,
                                  void *Data) {
  SymbolState *S = (SymbolState *)Data;

  prof_hsa_symbol_kind_t Kind;
  if (pHsaSymGetInfo(Sym, PROF_HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &Kind) !=
          PROF_HSA_STATUS_SUCCESS ||
      Kind != PROF_HSA_SYMBOL_KIND_VARIABLE)
    return PROF_HSA_STATUS_SUCCESS;

  uint32_t NameLen = 0;
  if (pHsaSymGetInfo(Sym, PROF_HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH,
                     &NameLen) != PROF_HSA_STATUS_SUCCESS ||
      NameLen != sizeof(ProfileSectionsSymbol) - 1)
    return PROF_HSA_STATUS_SUCCESS;

  char NameBuf[64];
  if (NameLen + 1 > sizeof(NameBuf))
    return PROF_HSA_STATUS_SUCCESS;
  if (pHsaSymGetInfo(Sym, PROF_HSA_EXECUTABLE_SYMBOL_INFO_NAME, NameBuf) !=
      PROF_HSA_STATUS_SUCCESS)
    return PROF_HSA_STATUS_SUCCESS;
  NameBuf[NameLen] = '\0';

  if (strcmp(NameBuf, ProfileSectionsSymbol) != 0)
    return PROF_HSA_STATUS_SUCCESS;

  uint64_t Addr = 0;
  if (pHsaSymGetInfo(Sym, PROF_HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS,
                     &Addr) != PROF_HSA_STATUS_SUCCESS ||
      Addr == 0) {
    if (isVerboseMode())
      PROF_NOTE("%s", "failed to read __llvm_profile_sections address\n");
    return PROF_HSA_STATUS_SUCCESS;
  }

  S->found++;

  // Read the bounds table first to dedup (and detect empty sections) before
  // the full copy/relocate done by processDeviceOffloadPrf.
  __llvm_profile_gpu_sections Sec;
  if (memcpyDeviceToHost(&Sec, (void *)(uintptr_t)Addr, sizeof(Sec)) != 0) {
    PROF_WARN("%s", "failed to copy device bounds table\n");
    return PROF_HSA_STATUS_SUCCESS;
  }
  if (profBoundsAlreadyDrained(Sec.DataStart, Sec.CountersStart,
                               Sec.NamesStart)) {
    if (isVerboseMode())
      PROF_NOTE("%s", "device bounds already drained, skipping\n");
    return PROF_HSA_STATUS_SUCCESS;
  }

  size_t DataBytes = (const char *)Sec.DataStop - (const char *)Sec.DataStart;
  size_t CntsBytes =
      (const char *)Sec.CountersStop - (const char *)Sec.CountersStart;
  if (DataBytes == 0 || CntsBytes == 0) {
    // Empty code object: nothing to write. Mark seen so we don't revisit it.
    profRecordDrainedBounds(Sec.DataStart, Sec.CountersStart, Sec.NamesStart);
    return PROF_HSA_STATUS_SUCCESS;
  }

  // Generate a collision-free target. Multiple distinct device code objects on
  // the same arch (e.g. non-RDC multi-TU) must not clobber each other's file.
  static int DrainIndex = 0;
  char Target[96];
  if (DrainIndex == 0)
    snprintf(Target, sizeof(Target), "%s", S->arch);
  else
    snprintf(Target, sizeof(Target), "%s.%d", S->arch, DrainIndex);

  // processDeviceOffloadPrf returns 0 on a successful write, -1 on error.
  // Record the bounds (and advance the target index) only on success so a
  // transient error stays retryable on a later agent or collect call.
  int Rc = processDeviceOffloadPrf((void *)(uintptr_t)Addr, Target, nullptr);
  if (Rc == 0) {
    S->drained++;
    DrainIndex++;
    profRecordDrainedBounds(Sec.DataStart, Sec.CountersStart, Sec.NamesStart);
  }

  return PROF_HSA_STATUS_SUCCESS;
}

static prof_hsa_status_t collectAgent(prof_hsa_agent_t Agent, void *Data) {
  prof_hsa_device_type_t DevType;
  if (pHsaAgentGetInfo(Agent, PROF_HSA_AGENT_INFO_DEVICE, &DevType) !=
          PROF_HSA_STATUS_SUCCESS ||
      DevType != PROF_HSA_DEVICE_TYPE_GPU)
    return PROF_HSA_STATUS_SUCCESS;

  WalkState *W = (WalkState *)Data;
  if (W->num_agents >= PROF_MAX_GPU_AGENTS)
    return PROF_HSA_STATUS_SUCCESS;

  GpuAgent &GA = W->agents[W->num_agents++];
  GA.agent = Agent;
  char Name[64];
  __builtin_memset(Name, 0, sizeof(Name));
  pHsaAgentGetInfo(Agent, PROF_HSA_AGENT_INFO_NAME, Name);
  size_t N = strnlen(Name, sizeof(GA.arch) - 1);
  __builtin_memcpy(GA.arch, Name, N);
  GA.arch[N] = '\0';
  if (!GA.arch[0])
    strncpy(GA.arch, "amdgpu", sizeof(GA.arch) - 1);

  if (isVerboseMode())
    PROF_NOTE("GPU agent %d: %s\n", W->num_agents - 1, GA.arch);
  return PROF_HSA_STATUS_SUCCESS;
}

/* Reentrancy guard and "drained data at least once" latch. The collect hook
 * may run more than once (an explicit early __llvm_profile_write_file plus the
 * exit write); a successful walk latches HsaDrainCompleted so we never re-emit
 * duplicate .profraw files, while transient no-op outcomes ("runtime not yet
 * loadable", "no GPU agents", "no loaded segments", "nothing instrumented")
 * stay retryable so a later call can still pick up code objects loaded later.
 * HsaDrainInProgress prevents a concurrent or reentrant call (e.g. a library
 * destructor) from corrupting the global SeenBounds table. Both flags use
 * acquire/release atomics. */
static int HsaDrainInProgress = 0;
static int HsaDrainCompleted = 0;

static int drainDevicesViaHsa(void) {
  if (__atomic_load_n(&HsaDrainCompleted, __ATOMIC_ACQUIRE))
    return 0;

  int Expected = 0;
  if (!__atomic_compare_exchange_n(&HsaDrainInProgress, &Expected, 1,
                                   /*weak=*/0, __ATOMIC_ACQ_REL,
                                   __ATOMIC_ACQUIRE))
    return 0;

  struct InProgressGuard {
    ~InProgressGuard() {
      __atomic_store_n(&HsaDrainInProgress, 0, __ATOMIC_RELEASE);
    }
  } _Guard;

  if (loadHsaRuntimePointers() != 0)
    return 0; /* Runtime unavailable: stay retryable. */

  WalkState W;
  __builtin_memset(&W, 0, sizeof(W));
  prof_hsa_status_t St = pHsaIterateAgents(collectAgent, &W);
  if (St != PROF_HSA_STATUS_SUCCESS && St != PROF_HSA_STATUS_INFO_BREAK) {
    PROF_WARN("hsa_iterate_agents failed (0x%x)\n", St);
    return -1;
  }
  if (W.num_agents == 0) {
    if (isVerboseMode())
      PROF_NOTE("%s", "no GPU agents present; nothing to drain (will retry)\n");
    return 0;
  }

  /* query_segment_descriptors ships in every loader-extension version and is
   * more permissive than iterate_executables on ROCm. It yields the loaded
   * (agent, executable) pairs directly. */
  size_t NumSegs = 0;
  St = pQuerySegDescs(nullptr, &NumSegs);
  if (St != PROF_HSA_STATUS_SUCCESS) {
    PROF_WARN("query_segment_descriptors(count) failed (0x%x)\n", St);
    return -1;
  }
  if (NumSegs == 0) {
    if (isVerboseMode())
      PROF_NOTE("%s", "no loaded segments; nothing to drain (will retry)\n");
    return 0;
  }

  prof_hsa_loader_segment_descriptor_t *Segs =
      (prof_hsa_loader_segment_descriptor_t *)calloc(NumSegs, sizeof(*Segs));
  if (!Segs) {
    PROF_ERR("%s\n", "failed to allocate segment descriptor array");
    return -1;
  }
  UniqueFree SegsOwner(Segs);

  St = pQuerySegDescs(Segs, &NumSegs);
  if (St != PROF_HSA_STATUS_SUCCESS) {
    PROF_WARN("query_segment_descriptors(fetch) failed (0x%x)\n", St);
    return -1;
  }

  if (isVerboseMode())
    PROF_NOTE("query_segment_descriptors: %zu segments\n", NumSegs);

  /* Walk unique (agent, executable) pairs. */
  enum { kMaxPairs = 512 };
  uint64_t SeenAgents[kMaxPairs];
  uint64_t SeenExecs[kMaxPairs];
  int NumPairs = 0;
  int IterFailures = 0;

  for (size_t i = 0; i < NumSegs; ++i) {
    if (Segs[i].executable.handle == 0 || Segs[i].agent.handle == 0)
      continue;

    int Seen = 0;
    for (int j = 0; j < NumPairs; ++j)
      if (SeenAgents[j] == Segs[i].agent.handle &&
          SeenExecs[j] == Segs[i].executable.handle) {
        Seen = 1;
        break;
      }
    if (Seen)
      continue;
    if (NumPairs < kMaxPairs) {
      SeenAgents[NumPairs] = Segs[i].agent.handle;
      SeenExecs[NumPairs] = Segs[i].executable.handle;
      NumPairs++;
    }

    const char *Arch = nullptr;
    for (int k = 0; k < W.num_agents; ++k)
      if (W.agents[k].agent.handle == Segs[i].agent.handle) {
        Arch = W.agents[k].arch;
        break;
      }
    if (!Arch)
      continue; /* not a GPU agent we collected */

    SymbolState S;
    __builtin_memset(&S, 0, sizeof(S));
    S.arch = Arch;
    if (isVerboseMode())
      PROF_NOTE("walking executable 0x%llx on %s\n",
                (unsigned long long)Segs[i].executable.handle, Arch);
    prof_hsa_status_t IterSt =
        pHsaExecIterAgentSyms(Segs[i].executable, Segs[i].agent, onSymbol, &S);
    if (IterSt != PROF_HSA_STATUS_SUCCESS &&
        IterSt != PROF_HSA_STATUS_INFO_BREAK) {
      PROF_WARN("hsa_executable_iterate_agent_symbols on executable 0x%llx "
                "failed (0x%x)\n",
                (unsigned long long)Segs[i].executable.handle, IterSt);
      IterFailures++;
    }
    W.total_found += S.found;
    W.total_drained += S.drained;
  }

  if (isVerboseMode())
    PROF_NOTE("HSA walk complete: agents=%d pairs=%d found=%d drained=%d "
              "iter-failures=%d\n",
              W.num_agents, NumPairs, W.total_found, W.total_drained,
              IterFailures);

  /* Latch only when we actually drained data. Deliberately do NOT latch the
   * "walked everything but found nothing new" case: an early collect call can
   * run before any kernel launch, and latching it would suppress the real
   * exit-time drain once kernels do run. Repeating a no-op walk is cheap. */
  if (W.total_drained > 0)
    __atomic_store_n(&HsaDrainCompleted, 1, __ATOMIC_RELEASE);
  return (IterFailures > 0) ? -1 : 0;
}

/* NOTE: deliberately no library constructor that calls hsa_init() here.
 * Bringing HSA up merely because the instrumented library was loaded poisons
 * fork-based callers: frameworks and tests (e.g. RCCL's unit tests) keep
 * HIP/HSA uninitialized in the parent and only touch HIP inside forked
 * children. A parent that has already hsa_init()'d makes those children crash
 * inside HSA (HSA state is not valid across fork()). HSA is instead brought up
 * lazily from drainDevicesViaHsa() -> loadHsaRuntimePointers(); see the init
 * rationale there. */

#endif /* defined(__linux__) && !defined(_WIN32) -- HSA drain */

/* -------------------------------------------------------------------------- */
/*  Collect device-side profile data                                          */
/* -------------------------------------------------------------------------- */

extern "C" int __llvm_profile_hip_collect_device_data(void) {
  int Ret = 0;

  /* Host-shadow drain: static-linked kernels (host __hipRegisterVar shadows)
   * and intercepted dynamic modules. Only meaningful when something registered
   * host-side; skipped entirely for pure device-linked programs (RCCL), which
   * the supplemental HSA pass below handles. */
  if ((NumShadowVariables != 0 || NumDynamicModules != 0) && isHipAvailable()) {
    /* Shadow variables (static-linked kernels): drain from every device. */
    if (NumShadowVariables > 0) {
      int OrigDevice = -1;
      hipGetDevice(&OrigDevice);

      for (int Dev = 0; Dev < NumDevices; ++Dev) {
        if (!shouldCollectDevice(Dev)) {
          if (isVerboseMode())
            PROF_NOTE("Skipping unused device %d\n", Dev);
          continue;
        }
#if defined(__linux__) && !defined(_WIN32)
        /* When no kernel launch was tracked at all, shouldCollectDevice()
         * falls back to collect-all, which can fault/hang reading a
         * non-resident device's sections on a multi-GPU host (e.g. a program
         * that never launches, collects before its first launch, or launches
         * only via an untracked API). On Linux the supplemental HSA drain
         * below covers those cases safely -- it walks only code objects
         * actually resident on each agent -- so skip the host-shadow pass
         * entirely rather than take the unsafe fallback. */
        if (!__atomic_load_n(&AnyDeviceUsed, __ATOMIC_ACQUIRE)) {
          if (isVerboseMode())
            PROF_NOTE("No tracked launch; deferring device %d to HSA drain\n",
                      Dev);
          continue;
        }
#endif
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
            snprintf(TargetWithIdx, sizeof(TargetWithIdx), "%s.%d", ArchName,
                     i);
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
  }

#if defined(__linux__) && !defined(_WIN32)
  /* Supplemental HSA-introspection drain: catches device code objects with no
   * host-side shadow (e.g. RCCL device-linked kernels). Runs after the
   * host-shadow drain so already-drained sections are deduped out, and runs
   * even when there are no host shadows at all (the common RCCL case). */
  if (drainDevicesViaHsa() != 0)
    Ret = -1;
#endif

  if (Ret != 0)
    PROF_WARN("%s\n", "failed to collect device profile data");
  return Ret;
}

/* Linux HIP interceptors. */

#if defined(__linux__) && !defined(_WIN32)

typedef struct {
  unsigned int x;
  unsigned int y;
  unsigned int z;
} HipDim3;

typedef struct {
  void *Func;
  HipDim3 GridDim;
  HipDim3 BlockDim;
  void **Args;
  size_t SharedMem;
  HipStream Stream;
} HipLaunchParams;

typedef struct {
  HipDim3 GridDim;
  HipDim3 BlockDim;
  size_t DynamicSmemBytes;
  HipStream Stream;
  void *Attrs;
  unsigned NumAttrs;
} HipLaunchConfig;

typedef void *HipFunction;
typedef void *HipEvent;
typedef void *HipGraphExec;

static int recordHipLaunchResult(int Rc, HipStream Stream) {
  if (Rc == 0)
    markLaunchStreamDeviceUsed(Stream);
  return Rc;
}

static int recordHipMultiDeviceLaunchResult(int Rc,
                                            HipLaunchParams *LaunchParams,
                                            int NumLaunches) {
  if (Rc != 0 || !LaunchParams || NumLaunches <= 0)
    return Rc;
  for (int I = 0; I < NumLaunches; ++I)
    markLaunchStreamDeviceUsed(LaunchParams[I].Stream);
  return Rc;
}

INTERCEPTOR(int, hipLaunchKernel, const void *Function, HipDim3 GridDim,
            HipDim3 BlockDim, void **Args, size_t SharedMemBytes,
            HipStream Stream) {
  return recordHipLaunchResult(REAL(hipLaunchKernel)(Function, GridDim,
                                                     BlockDim, Args,
                                                     SharedMemBytes, Stream),
                               Stream);
}

INTERCEPTOR(int, hipLaunchKernel_spt, const void *Function, HipDim3 GridDim,
            HipDim3 BlockDim, void **Args, size_t SharedMemBytes,
            HipStream Stream) {
  return recordHipLaunchResult(
      REAL(hipLaunchKernel_spt)(Function, GridDim, BlockDim, Args,
                                SharedMemBytes, Stream),
      Stream);
}

INTERCEPTOR(int, hipExtLaunchKernel, const void *Function, HipDim3 GridDim,
            HipDim3 BlockDim, void **Args, size_t SharedMemBytes,
            HipStream Stream, HipEvent StartEvent, HipEvent StopEvent,
            int Flags) {
  return recordHipLaunchResult(
      REAL(hipExtLaunchKernel)(Function, GridDim, BlockDim, Args,
                               SharedMemBytes, Stream, StartEvent, StopEvent,
                               Flags),
      Stream);
}

INTERCEPTOR(int, hipLaunchKernelExC, const HipLaunchConfig *Config,
            const void *Function, void **Args) {
  int Rc = REAL(hipLaunchKernelExC)(Config, Function, Args);
  return recordHipLaunchResult(Rc, Config ? Config->Stream : nullptr);
}

INTERCEPTOR(int, hipLaunchCooperativeKernel, const void *Function,
            HipDim3 GridDim, HipDim3 BlockDim, void **KernelParams,
            unsigned SharedMemBytes, HipStream Stream) {
  return recordHipLaunchResult(
      REAL(hipLaunchCooperativeKernel)(Function, GridDim, BlockDim,
                                       KernelParams, SharedMemBytes, Stream),
      Stream);
}

INTERCEPTOR(int, hipLaunchCooperativeKernel_spt, const void *Function,
            HipDim3 GridDim, HipDim3 BlockDim, void **KernelParams,
            unsigned SharedMemBytes, HipStream Stream) {
  return recordHipLaunchResult(
      REAL(hipLaunchCooperativeKernel_spt)(
          Function, GridDim, BlockDim, KernelParams, SharedMemBytes, Stream),
      Stream);
}

INTERCEPTOR(int, hipLaunchCooperativeKernelMultiDevice,
            HipLaunchParams *LaunchParams, int NumDevices, unsigned Flags) {
  return recordHipMultiDeviceLaunchResult(
      REAL(hipLaunchCooperativeKernelMultiDevice)(LaunchParams, NumDevices,
                                                  Flags),
      LaunchParams, NumDevices);
}

INTERCEPTOR(int, hipExtLaunchMultiKernelMultiDevice,
            HipLaunchParams *LaunchParams, int NumDevices, unsigned Flags) {
  return recordHipMultiDeviceLaunchResult(
      REAL(hipExtLaunchMultiKernelMultiDevice)(LaunchParams, NumDevices, Flags),
      LaunchParams, NumDevices);
}

INTERCEPTOR(int, hipModuleLaunchKernel, HipFunction Function, unsigned GridDimX,
            unsigned GridDimY, unsigned GridDimZ, unsigned BlockDimX,
            unsigned BlockDimY, unsigned BlockDimZ, unsigned SharedMemBytes,
            HipStream Stream, void **KernelParams, void **Extra) {
  return recordHipLaunchResult(
      REAL(hipModuleLaunchKernel)(Function, GridDimX, GridDimY, GridDimZ,
                                  BlockDimX, BlockDimY, BlockDimZ,
                                  SharedMemBytes, Stream, KernelParams, Extra),
      Stream);
}

INTERCEPTOR(int, hipExtModuleLaunchKernel, HipFunction Function,
            unsigned GridDimX, unsigned GridDimY, unsigned GridDimZ,
            unsigned BlockDimX, unsigned BlockDimY, unsigned BlockDimZ,
            size_t SharedMemBytes, HipStream Stream, void **KernelParams,
            void **Extra, HipEvent StartEvent, HipEvent StopEvent,
            unsigned Flags) {
  return recordHipLaunchResult(
      REAL(hipExtModuleLaunchKernel)(Function, GridDimX, GridDimY, GridDimZ,
                                     BlockDimX, BlockDimY, BlockDimZ,
                                     SharedMemBytes, Stream, KernelParams,
                                     Extra, StartEvent, StopEvent, Flags),
      Stream);
}

INTERCEPTOR(int, hipGraphLaunch, HipGraphExec GraphExec, HipStream Stream) {
  return recordHipLaunchResult(REAL(hipGraphLaunch)(GraphExec, Stream), Stream);
}

INTERCEPTOR(int, hipGraphLaunch_spt, HipGraphExec GraphExec, HipStream Stream) {
  return recordHipLaunchResult(REAL(hipGraphLaunch_spt)(GraphExec, Stream),
                               Stream);
}

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

__attribute__((constructor)) static void installHipInterceptors() {
  /* Avoid interception unless the HIP runtime is already loaded. */
  int HasModuleLoad = dlsym(RTLD_DEFAULT, "hipModuleLoad") != nullptr;
  int InstalledLaunch = 0;
#define TRY_INTERCEPT_LAUNCH(Name)                                             \
  do {                                                                         \
    if (dlsym(RTLD_DEFAULT, #Name))                                            \
      InstalledLaunch |= INTERCEPT_FUNCTION(Name);                             \
  } while (0)
  TRY_INTERCEPT_LAUNCH(hipLaunchKernel);
  TRY_INTERCEPT_LAUNCH(hipLaunchKernel_spt);
  TRY_INTERCEPT_LAUNCH(hipExtLaunchKernel);
  TRY_INTERCEPT_LAUNCH(hipLaunchKernelExC);
  TRY_INTERCEPT_LAUNCH(hipLaunchCooperativeKernel);
  TRY_INTERCEPT_LAUNCH(hipLaunchCooperativeKernel_spt);
  TRY_INTERCEPT_LAUNCH(hipLaunchCooperativeKernelMultiDevice);
  TRY_INTERCEPT_LAUNCH(hipExtLaunchMultiKernelMultiDevice);
  TRY_INTERCEPT_LAUNCH(hipModuleLaunchKernel);
  TRY_INTERCEPT_LAUNCH(hipExtModuleLaunchKernel);
  TRY_INTERCEPT_LAUNCH(hipGraphLaunch);
  TRY_INTERCEPT_LAUNCH(hipGraphLaunch_spt);
#undef TRY_INTERCEPT_LAUNCH
  int InstalledAny = InstalledLaunch;
  if (HasModuleLoad) {
    HasModuleLoad = INTERCEPT_FUNCTION(hipModuleLoad);
    InstalledAny |= HasModuleLoad;
  }
  if (!InstalledAny)
    return;
  if (isVerboseMode())
    PROF_NOTE("%s", "Installing HIP interceptors\n");
  if (HasModuleLoad) {
    INTERCEPT_FUNCTION(hipModuleLoadData);
    INTERCEPT_FUNCTION(hipModuleLoadDataEx);
    INTERCEPT_FUNCTION(hipModuleUnload);
  }
}

#endif /* __linux__ */
