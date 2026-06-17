//===- InstrProfilingPlatformROCmHSA.cpp - ROCm HSA device drain ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Supplemental HSA-introspection drain (Linux only).
//
// The host-shadow drain in InstrProfilingPlatformROCm.cpp only sees device
// code objects registered host-side (__hipRegisterVar shadows) or loaded
// through an intercepted hipModuleLoad* call. Device code linked by the offload
// device linker with no host-side shadow -- e.g. RCCL, whose many device
// functions are glued into a single kernel with no source module -- is
// invisible to it. This pass walks every GPU agent's loaded executables via
// HSA, finds each __llvm_profile_sections table directly on the device, and
// drains the ones the host-shadow pass did not already handle (deduped by the
// device section-bounds tuple). It reuses processDeviceOffloadPrf() for the
// copy/relocate/write so the on-disk profraw layout is identical.
//
// There is deliberately no Windows counterpart: HSA introspection is Linux-only
// and Windows relies entirely on the host-shadow HIP drain. On any non-Linux
// target this file compiles to an empty translation unit.
//
//===----------------------------------------------------------------------===//

#if defined(__linux__) && !defined(_WIN32)

extern "C" {
#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include "InstrProfilingPort.h"
}

#include "InstrProfilingPlatformROCmInternal.h"
#include "interception/interception.h"
// C library headers (not <cstdio> etc.): clang_rt.profile is built with
// -nostdinc++ and avoids the C++ standard library (see profile/CMakeLists.txt).
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace __prof_rocm;

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
  if (!hipMemcpyAvailable()) {
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

/* Grown on demand (doubling) rather than fixed-cap: in non-RDC mode the entry
 * count scales like num_code_objects * num_agents, so any fixed cap could be
 * exceeded and silently lose dedup coverage (double-counting drained sections).
 * Starts at PROF_SEEN_BOUNDS_INIT_CAP. */
#define PROF_SEEN_BOUNDS_INIT_CAP 64
static ProfBoundsTuple *SeenBounds = nullptr;
static int NumSeenBounds = 0;
static int CapSeenBounds = 0;

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
void __prof_rocm::profRecordDrainedBounds(const void *D, const void *C,
                                          const void *N) {
  if (profBoundsAlreadyDrained(D, C, N))
    return;
  if (NumSeenBounds == CapSeenBounds) {
    int NewCap = CapSeenBounds ? CapSeenBounds * 2 : PROF_SEEN_BOUNDS_INIT_CAP;
    ProfBoundsTuple *New =
        (ProfBoundsTuple *)realloc(SeenBounds, NewCap * sizeof(*New));
    /* Best-effort: on OOM keep the existing table and skip recording. The
     * worst case is that this one section is drained again later (a duplicate
     * profraw record), never a crash. */
    if (!New)
      return;
    SeenBounds = New;
    CapSeenBounds = NewCap;
  }
  SeenBounds[NumSeenBounds].data = D;
  SeenBounds[NumSeenBounds].cnts = C;
  SeenBounds[NumSeenBounds].names = N;
  NumSeenBounds++;
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

int __prof_rocm::drainDevicesViaHsa(void) {
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
