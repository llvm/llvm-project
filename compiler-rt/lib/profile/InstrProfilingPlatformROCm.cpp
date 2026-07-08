//===- InstrProfilingPlatformROCm.cpp - Profile data ROCm platform -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

extern "C" {
#include "InstrProfiling.h"
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

#include "InstrProfilingPlatformROCmInternal.h"

// shortcut to shared helper names
using namespace __prof_rocm;

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

int __prof_rocm::isVerboseMode() {
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

void __prof_rocm::ensureHipLoaded(void) {
#ifdef _WIN32
  InitOnceExecuteOnce(&HipLoadedOnce, ensureHipLoadedCb, NULL, NULL);
#else
  pthread_once(&HipLoadedOnce, doEnsureHipLoaded);
#endif
}

// Accessor for the HSA drain: true once the loaded HIP runtime exposes
// hipMemcpy. Kept here so pHipMemcpy stays file-private to this TU.
int __prof_rocm::hipMemcpyAvailable() { return pHipMemcpy != nullptr; }

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
int __prof_rocm::memcpyDeviceToHost(void *Dst, const void *Src, size_t Size) {
  return hipMemcpy(Dst, Src, Size, 2 /* DToH */);
}

[[maybe_unused]]
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

  if (growArray((void **)&MI->TUs, &MI->CapTUs, MI->NumTUs + 1, 4,
                sizeof(*MI->TUs))) {
    PROF_ERR("%s\n", "failed to grow TU array");
    return 0;
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

  if (growArray((void **)&DynamicModules, &CapDynamicModules,
                NumDynamicModules + 1, 64, sizeof(*DynamicModules))) {
    unlockDynamicModules();
    return;
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

static void **OffloadShadowVariables = nullptr;
static int NumShadowVariables = 0;
static int CapShadowVariables = 0;

struct OffloadSectionShadow {
  void *Data;
  void *Counters;
  void *UniformCounters;
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
  return growArray((void **)&OffloadSectionShadowGroups,
                   &CapSectionShadowGroups, CapShadowVariables,
                   CapShadowVariables, sizeof(*OffloadSectionShadowGroups));
}

static int ensureSectionShadowCapacity(OffloadSectionShadowGroup *Group,
                                       int MinCapacity) {
  return growArray((void **)&Group->Shadows, &Group->CapShadows, MinCapacity, 4,
                   sizeof(*Group->Shadows));
}

extern "C" void __llvm_profile_offload_register_shadow_variable(void *ptr) {
  if (growArray((void **)&OffloadShadowVariables, &CapShadowVariables,
                NumShadowVariables + 1, 64, sizeof(*OffloadShadowVariables)))
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

  /* Match CGCUDANV.cpp: data, counters, uniform counters, then names for each
   * kernel. */
  OffloadSectionShadowGroup *Group =
      &OffloadSectionShadowGroups[NumShadowVariables - 1];
  int ShadowIndex = Group->NumSections / 4;
  if (ensureSectionShadowCapacity(Group, ShadowIndex + 1))
    return;
  if (ShadowIndex >= Group->NumShadows)
    Group->NumShadows = ShadowIndex + 1;

  OffloadSectionShadow *Shadow = &Group->Shadows[ShadowIndex];
  switch (Group->NumSections % 4) {
  case 0:
    Shadow->Data = ptr;
    break;
  case 1:
    Shadow->Counters = ptr;
    break;
  case 2:
    Shadow->UniformCounters = ptr;
    break;
  case 3:
    Shadow->Names = ptr;
    break;
  }
  ++Group->NumSections;
}

namespace {

struct ProfileSectionCopy {
  const char *Name;
  const void *DevBegin;
  size_t Size;
  const void *&CachedDevBegin;
  char *&CachedHost;
  size_t &CachedSize;
  UniqueFree Owner;
  char *HostBegin = nullptr;
  bool Reused = false;

  ProfileSectionCopy(const char *Name, const void *DevBegin, size_t Size,
                     const void *&CachedDevBegin, char *&CachedHost,
                     size_t &CachedSize)
      : Name(Name), DevBegin(DevBegin), Size(Size),
        CachedDevBegin(CachedDevBegin), CachedHost(CachedHost),
        CachedSize(CachedSize) {}

  ProfileSectionCopy(const ProfileSectionCopy &) = delete;
  ProfileSectionCopy &operator=(const ProfileSectionCopy &) = delete;

  int prepare() {
    if (Size == 0)
      return 0;
    if (DevBegin == CachedDevBegin && Size == CachedSize) {
      HostBegin = CachedHost;
      Reused = true;
      if (isVerboseMode())
        PROF_NOTE("Reusing cached %s section (%zu bytes)\n", Name, Size);
    } else {
      HostBegin = static_cast<char *>(malloc(Size));
      Owner.reset(HostBegin);
    }
    return HostBegin ? 0 : -1;
  }

  int copy() {
    if (Size == 0 || Reused)
      return 0;
    return memcpyDeviceToHost(HostBegin, DevBegin, Size);
  }

  void commitCache() {
    if (Reused || Size == 0)
      return;
    CachedDevBegin = DevBegin;
    CachedHost = HostBegin;
    CachedSize = Size;
    Owner.release();
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
  const void *UniformCounters;
  const void *Names;
  size_t DataSize;
  size_t CountersSize;
  size_t UniformCountersSize;
  size_t NamesSize;
  size_t DataOffset;
  size_t CountersOffset;
  size_t UniformCountersOffset;
  size_t NamesOffset;
};

static int
hasCompleteSectionShadows(const OffloadSectionShadowGroup *Sections) {
  if (!Sections || Sections->NumShadows == 0 || Sections->NumSections % 4 != 0)
    return 0;
  for (int I = 0; I < Sections->NumShadows; ++I) {
    if (!Sections->Shadows[I].Data || !Sections->Shadows[I].Counters ||
        !Sections->Shadows[I].UniformCounters || !Sections->Shadows[I].Names)
      return 0;
  }
  return 1;
}

int __prof_rocm::processDeviceOffloadPrf(
    void *DeviceOffloadPrf, const char *Target,
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
  const void *DevUniformCntsBegin = HostSections.UniformCountersStart;
  const void *DevCntsEnd = HostSections.CountersStop;
  const void *DevDataEnd = HostSections.DataStop;
  const void *DevNamesEnd = HostSections.NamesStop;
  const void *DevUniformCntsEnd = HostSections.UniformCountersStop;

  size_t CountersSize = (const char *)DevCntsEnd - (const char *)DevCntsBegin;
  size_t DataSize = (const char *)DevDataEnd - (const char *)DevDataBegin;
  size_t NamesSize = (const char *)DevNamesEnd - (const char *)DevNamesBegin;
  size_t UniformCountersSize =
      (const char *)DevUniformCntsEnd - (const char *)DevUniformCntsBegin;

  int UseRegisteredSections = hasCompleteSectionShadows(Sections);
  RegisteredSectionRange *RegisteredRanges = nullptr;
  int NumRegisteredRanges = 0;

  if (isVerboseMode())
    PROF_NOTE("Section pointers: Cnts=[%p,%p]=%zu Data=[%p,%p]=%zu "
              "Names=[%p,%p]=%zu UCnts=[%p,%p]=%zu\n",
              DevCntsBegin, DevCntsEnd, CountersSize, DevDataBegin, DevDataEnd,
              DataSize, DevNamesBegin, DevNamesEnd, NamesSize,
              DevUniformCntsBegin, DevUniformCntsEnd, UniformCountersSize);

  if (CountersSize == 0 || DataSize == 0)
    return 0;

  int ret = -1;

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

  static const void *CachedDevUCntsBegin = nullptr;
  static char *CachedHostUCnts = nullptr;
  static size_t CachedUCntsSize = 0;

  ProfileSectionCopy Cnts("counters", DevCntsBegin, CountersSize,
                          CachedDevCntsBegin, CachedHostCnts, CachedCntsSize);
  ProfileSectionCopy Data("data", DevDataBegin, DataSize, CachedDevDataBegin,
                          CachedHostData, CachedDataSize);
  ProfileSectionCopy Names("names", DevNamesBegin, NamesSize,
                           CachedDevNamesBegin, CachedHostNames,
                           CachedNamesSize);
  ProfileSectionCopy UCnts("ucnts", DevUniformCntsBegin, UniformCountersSize,
                           CachedDevUCntsBegin, CachedHostUCnts,
                           CachedUCntsSize);

  UniqueFree RegisteredRangeOwner;

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
    size_t RegisteredUniformCountersSize = 0;
    size_t RegisteredNamesSize = 0;
    for (int I = 0; I < NumRegisteredRanges; ++I) {
      void *Data = nullptr;
      void *Counters = nullptr;
      void *UniformCounters = nullptr;
      void *Names = nullptr;
      size_t ThisDataSize = 0;
      size_t ThisCountersSize = 0;
      size_t ThisUniformCountersSize = 0;
      size_t ThisNamesSize = 0;
      OffloadSectionShadow *Shadow = &Sections->Shadows[I];
      if (getRegisteredSectionBounds(Shadow->Data, &Data, &ThisDataSize) != 0 ||
          getRegisteredSectionBounds(Shadow->Counters, &Counters,
                                     &ThisCountersSize) != 0 ||
          getRegisteredSectionBounds(Shadow->UniformCounters, &UniformCounters,
                                     &ThisUniformCountersSize) != 0 ||
          getRegisteredSectionBounds(Shadow->Names, &Names, &ThisNamesSize) !=
              0) {
        PROF_ERR("%s\n", "failed to get registered section bounds");
        return -1;
      }

      RegisteredRanges[I].Data = Data;
      RegisteredRanges[I].Counters = Counters;
      RegisteredRanges[I].UniformCounters = UniformCounters;
      RegisteredRanges[I].Names = Names;
      RegisteredRanges[I].DataSize = ThisDataSize;
      RegisteredRanges[I].CountersSize = ThisCountersSize;
      RegisteredRanges[I].UniformCountersSize = ThisUniformCountersSize;
      RegisteredRanges[I].NamesSize = ThisNamesSize;
      RegisteredRanges[I].DataOffset = RegisteredDataSize;
      RegisteredRanges[I].CountersOffset = RegisteredCountersSize;
      RegisteredRanges[I].UniformCountersOffset = RegisteredUniformCountersSize;
      RegisteredDataSize += ThisDataSize;
      RegisteredCountersSize += ThisCountersSize;
      RegisteredUniformCountersSize += ThisUniformCountersSize;

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
    UniformCountersSize = RegisteredUniformCountersSize;
    NamesSize = RegisteredNamesSize;
    Data.HostBegin = DataSize ? (char *)malloc(DataSize) : nullptr;
    Cnts.HostBegin = CountersSize ? (char *)malloc(CountersSize) : nullptr;
    UCnts.HostBegin =
        UniformCountersSize ? (char *)malloc(UniformCountersSize) : nullptr;
    Names.HostBegin = NamesSize ? (char *)malloc(NamesSize) : nullptr;
    Data.Owner.reset(Data.HostBegin);
    Cnts.Owner.reset(Cnts.HostBegin);
    UCnts.Owner.reset(UCnts.HostBegin);
    Names.Owner.reset(Names.HostBegin);
    if ((DataSize > 0 && !Data.HostBegin) ||
        (CountersSize > 0 && !Cnts.HostBegin) ||
        (UniformCountersSize > 0 && !UCnts.HostBegin) ||
        (NamesSize > 0 && !Names.HostBegin)) {
      PROF_ERR("%s\n", "failed to allocate host memory for device sections");
      return -1;
    }

    for (int I = 0; I < NumRegisteredRanges; ++I) {
      RegisteredSectionRange *R = &RegisteredRanges[I];
      if (memcpyDeviceToHost(Data.HostBegin + R->DataOffset, R->Data,
                             R->DataSize) != 0 ||
          memcpyDeviceToHost(Cnts.HostBegin + R->CountersOffset, R->Counters,
                             R->CountersSize) != 0 ||
          memcpyDeviceToHost(UCnts.HostBegin + R->UniformCountersOffset,
                             R->UniformCounters, R->UniformCountersSize) != 0) {
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
          memcpyDeviceToHost(Names.HostBegin + R->NamesOffset, R->Names,
                             R->NamesSize) != 0) {
        PROF_ERR("%s\n", "failed to copy profile sections from device");
        return -1;
      }
    }
  } else {
    if (Cnts.prepare() != 0 || Data.prepare() != 0 || Names.prepare() != 0 ||
        UCnts.prepare() != 0) {
      PROF_ERR("%s\n", "failed to allocate host memory for device sections");
      return -1;
    }

    if (Data.copy() != 0 || Cnts.copy() != 0 || Names.copy() != 0 ||
        UCnts.copy() != 0) {
      PROF_ERR("%s\n", "failed to copy profile sections from device");
      return -1;
    }

    /* Cache buffers so RDC-mode multi-shadow drains can reuse them.
     * release() prevents the scope guards from freeing what the cache owns. */
    Cnts.commitCache();
    Data.commitCache();
    Names.commitCache();
    UCnts.commitCache();
  }

  if (isVerboseMode())
    PROF_NOTE("Copied device sections: Counters=%zu, Data=%zu, Names=%zu, "
              "UniformCounters=%zu\n",
              CountersSize, DataSize, NamesSize, UniformCountersSize);

  // Arrange buffer as [Data][Padding][Counters][Names] to match the layout
  // expected by lprofWriteDataImpl (CountersDelta = CountersBegin - DataBegin).
  const uint64_t NumData = DataSize / sizeof(__llvm_profile_data);
  const uint64_t NumBitmapBytes = 0;
  const uint64_t NumUniformCounters = UniformCountersSize / sizeof(uint64_t);
  const uint64_t VTableSectionSize = 0;
  const uint64_t VNamesSize = 0;
  uint64_t PaddingBytesBeforeCounters, PaddingBytesAfterCounters,
      PaddingBytesAfterBitmapBytes, PaddingBytesAfterUniformCounters,
      PaddingBytesAfterNames, PaddingBytesAfterVTable, PaddingBytesAfterVNames;

  if (__llvm_profile_get_padding_sizes_for_counters(
          DataSize, CountersSize, NumBitmapBytes, NumUniformCounters, NamesSize,
          VTableSectionSize, VNamesSize, &PaddingBytesBeforeCounters,
          &PaddingBytesAfterCounters, &PaddingBytesAfterBitmapBytes,
          &PaddingBytesAfterUniformCounters, &PaddingBytesAfterNames,
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

  __builtin_memcpy(BufDataBegin, Data.HostBegin, DataSize);
  __builtin_memcpy(BufCountersBegin, Cnts.HostBegin, CountersSize);
  __builtin_memcpy(BufNamesBegin, Names.HostBegin, NamesSize);

  // CounterPtr and UniformCounterPtr are device-relative offsets; relocate
  // them for the file layout where the Data section precedes the Counters and
  // UniformCounters sections. Uniform counters are copied in linker (section)
  // order and located via their relative pointer, exactly like the regular
  // counters: llvm-profdata reads them through UniformCounterPtr (decrementing
  // UniformCountersDelta per record, just like CountersDelta) and does not
  // assume data-record order, so no reordering is needed.
  ptrdiff_t UCFileOffset = DataSize + PaddingBytesBeforeCounters +
                           CountersSize + PaddingBytesAfterCounters +
                           NumBitmapBytes + PaddingBytesAfterBitmapBytes;
  __llvm_profile_data *RelocatedData = (__llvm_profile_data *)BufDataBegin;
  for (uint64_t i = 0; i < NumData; ++i) {
    size_t DataRecordOffset = i * sizeof(__llvm_profile_data);
    const char *RangeDevDataBegin = (const char *)DevDataBegin;
    const char *RangeDevCountersBegin = (const char *)DevCntsBegin;
    const char *RangeDevUCntsBegin = (const char *)DevUniformCntsBegin;
    size_t RangeCountersOffset = 0;
    size_t RangeUCntsOffset = 0;
    if (UseRegisteredSections) {
      int FoundRange = 0;
      for (int R = 0; R < NumRegisteredRanges; ++R) {
        RegisteredSectionRange *Range = &RegisteredRanges[R];
        if (DataRecordOffset < Range->DataOffset ||
            DataRecordOffset >= Range->DataOffset + Range->DataSize)
          continue;
        RangeDevDataBegin = (const char *)Range->Data;
        RangeDevCountersBegin = (const char *)Range->Counters;
        RangeDevUCntsBegin = (const char *)Range->UniformCounters;
        RangeCountersOffset = Range->CountersOffset;
        RangeUCntsOffset = Range->UniformCountersOffset;
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
    if (RelocatedData[i].CounterPtr) {
      const char *DeviceCountersAddr =
          DeviceDataStructAddr + (ptrdiff_t)RelocatedData[i].CounterPtr;
      ptrdiff_t OffsetIntoCountersSection =
          DeviceCountersAddr - RangeDevCountersBegin;
      ptrdiff_t NewRelativeOffset =
          DataSize + PaddingBytesBeforeCounters + RangeCountersOffset +
          OffsetIntoCountersSection - (i * sizeof(__llvm_profile_data));
      __builtin_memcpy((char *)RelocatedData + i * sizeof(__llvm_profile_data) +
                           offsetof(__llvm_profile_data, CounterPtr),
                       &NewRelativeOffset, sizeof(NewRelativeOffset));
    }
    if (UCnts.HostBegin && RelocatedData[i].UniformCounterPtr) {
      const char *DeviceUCAddr =
          DeviceDataStructAddr + (ptrdiff_t)RelocatedData[i].UniformCounterPtr;
      ptrdiff_t OffsetIntoUCSection = DeviceUCAddr - RangeDevUCntsBegin;
      ptrdiff_t NewUCRelativeOffset = UCFileOffset + RangeUCntsOffset +
                                      OffsetIntoUCSection -
                                      (i * sizeof(__llvm_profile_data));
      __builtin_memcpy((char *)RelocatedData + i * sizeof(__llvm_profile_data) +
                           offsetof(__llvm_profile_data, UniformCounterPtr),
                       &NewUCRelativeOffset, sizeof(NewUCRelativeOffset));
    } else {
      __builtin_memset((char *)RelocatedData + i * sizeof(__llvm_profile_data) +
                           offsetof(__llvm_profile_data, UniformCounterPtr),
                       0, sizeof(RelocatedData[i].UniformCounterPtr));
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
      BufCountersBegin + CountersSize, UCnts.HostBegin,
      UCnts.HostBegin ? UCnts.HostBegin + UniformCountersSize : nullptr,
      BufNamesBegin, BufNamesBegin + NamesSize, nullptr);

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

/* -------------------------------------------------------------------------- */
/*  Collect device-side profile data                                          */
/* -------------------------------------------------------------------------- */

/* Host-shadow drain: static-linked kernels (host __hipRegisterVar shadows) and
 * intercepted dynamic modules. The caller gates this on
 * (NumShadowVariables || NumDynamicModules) && isHipAvailable(); pure
 * device-linked programs (RCCL) are handled by the supplemental HSA pass. */
static int collectHostShadowData(void) {
  int Ret = 0;

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
      /* When no kernel launch was tracked at all, shouldCollectDevice() falls
       * back to collect-all, which can fault/hang reading a non-resident
       * device's sections on a multi-GPU host. On Linux the supplemental HSA
       * drain covers those cases safely. */
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
        /* Stable name per shadow so a repeated drain (explicit collect plus the
         * atexit drain) overwrites its own profraw rather than emitting a
         * second one: bare arch for a single TU, arch.<i> for RDC multi-TU. */
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

  return Ret;
}

extern "C" int __llvm_profile_hip_collect_device_data(void) {
  int Ret = 0;

  if ((NumShadowVariables != 0 || NumDynamicModules != 0) && isHipAvailable() &&
      collectHostShadowData() != 0)
    Ret = -1;

#if defined(__linux__) && !defined(_WIN32)
  /* Supplemental HSA-introspection drain */
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

// interceptors must have external linkage
// NOLINTBEGIN(misc-use-internal-linkage)
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
// NOLINTEND(misc-use-internal-linkage)

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
