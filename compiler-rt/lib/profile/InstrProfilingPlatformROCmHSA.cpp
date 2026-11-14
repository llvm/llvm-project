//===- InstrProfilingPlatformROCmHSA.cpp - ROCm HSA device drain ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HSA resident-image drain (Linux only).
//
// This pass walks loaded executables on each GPU agent and drains their
// __llvm_profile_sections tables. It covers static HIP images and device-linked
// programs without host shadows, while avoiding lookups that would load unused
// static images. It shares processDeviceOffloadPrf() and section-bounds dedup
// with the dynamic-module drain.
//
//===----------------------------------------------------------------------===//

#if defined(__linux__)

extern "C" {
#include "InstrProfiling.h"
#include "InstrProfilingPort.h"
}

#include "InstrProfilingPlatformROCmInternal.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_offload_hsa.h"
// C (not C++) headers: clang_rt.profile is built -nostdinc++.
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace __prof_rocm;

/* Status-check shorthands, in the spirit of the thin HIP wrappers in
 * InstrProfilingPlatformROCm.cpp: every HSA entry point returns
 * hsa_status_t. hsaOkOrBreak() also accepts INFO_BREAK, which the
 * iterate_* callbacks use to stop early and is not an error. */
static inline bool hsaOk(hsa_status_t St) { return St == HSA_STATUS_SUCCESS; }
static inline bool hsaOkOrBreak(hsa_status_t St) {
  return St == HSA_STATUS_SUCCESS || St == HSA_STATUS_INFO_BREAK;
}

/* 0 = not attempted, 1 = ready, -1 = unavailable. Release publishes the HSA
 * pointers to a thread that observes ready. */
static int HsaRuntimeState = 0;

typedef decltype(&hsa_iterate_agents) hsa_iterate_agents_ty;
typedef decltype(&hsa_agent_get_info) hsa_agent_get_info_ty;
typedef decltype(&hsa_executable_iterate_agent_symbols)
    hsa_executable_iterate_agent_symbols_ty;
typedef decltype(&hsa_executable_symbol_get_info)
    hsa_executable_symbol_get_info_ty;
typedef decltype(LoaderApi::QuerySegmentDescriptors) hsa_query_segments_ty;

static hsa_iterate_agents_ty pHsaIterateAgents = nullptr;
static hsa_agent_get_info_ty pHsaAgentGetInfo = nullptr;
static hsa_executable_iterate_agent_symbols_ty pHsaExecIterAgentSyms = nullptr;
static hsa_executable_symbol_get_info_ty pHsaSymGetInfo = nullptr;
static hsa_query_segments_ty pQuerySegDescs = nullptr;

static int setHsaRuntimeState(int S) {
  __atomic_store_n(&HsaRuntimeState, S, __ATOMIC_RELEASE);
  return S > 0 ? 0 : -1;
}

/* Dlopen libhsa and call its hsa_init, then keep the loader slot this drain
 * calls. Runs on the first drain, never from a constructor -- see the
 * fork-safety note at end of file. */
static int bindHsaRuntime(void) {
  int State = __atomic_load_n(&HsaRuntimeState, __ATOMIC_ACQUIRE);
  if (State)
    return State > 0 ? 0 : -1;

  if (!__interception::DynamicLoaderAvailable()) {
    if (isVerboseMode())
      PROF_NOTE("%s", "Dynamic library loading not available - "
                      "HSA device profiling disabled\n");
    return setHsaRuntimeState(-1);
  }

  void *Hsa = __interception::OpenLibrary(SANITIZER_HSA_LIBRARY ".so");
  if (!Hsa)
    Hsa = __interception::OpenLibrary(SANITIZER_HSA_LIBRARY ".so.1");
  if (!Hsa) {
    if (isVerboseMode())
      PROF_NOTE("%s", "libhsa-runtime64.so not loadable - "
                      "HSA device profiling disabled\n");
    return setHsaRuntimeState(-1);
  }

  auto *pHsaInit =
      (decltype(&hsa_init))__interception::LookupSymbol(Hsa, "hsa_init");
  auto *pGetExtTable = (decltype(&hsa_system_get_major_extension_table))
      __interception::LookupSymbol(Hsa, "hsa_system_get_major_extension_table");
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

  hsa_status_t St = pHsaInit();
  if (!hsaOkOrBreak(St)) {
    if (isVerboseMode())
      PROF_NOTE("hsa_init failed (0x%x) - HSA device profiling disabled\n", St);
    return setHsaRuntimeState(-1);
  }

  // Full 1.04 table. A shorter runtime may succeed and leave the tail null.
  LoaderApi LoaderTable = {};
  St = pGetExtTable(HSA_EXTENSION_AMD_LOADER, 1, sizeof(LoaderTable),
                    &LoaderTable);
  if (!hsaOk(St) || !LoaderTable.QuerySegmentDescriptors) {
    PROF_WARN("AMD loader extension unavailable (0x%x) - "
              "HSA device profiling disabled\n",
              St);
    return setHsaRuntimeState(-1);
  }
  pQuerySegDescs = LoaderTable.QuerySegmentDescriptors;

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

int __prof_rocm::hsaRuntimeAvailable(void) { return bindHsaRuntime() == 0; }

/* The canonical device bounds-table symbol from InstrProfilingPlatformGPU.c. */
static const char ProfileSectionsSymbol[] = "__llvm_profile_sections";

/* Dedup of drained section-bounds tuples, shared with the host-shadow path
 * (processDeviceOffloadPrf records here on every successful drain) so each
 * unique counter set is drained exactly once across both paths.
 */
static ProfBoundsSet SeenBounds;

/* Has this bounds tuple already been drained? Pure check, no state mutation. */
static int profBoundsAlreadyDrained(const void *D, const void *C,
                                    const void *N) {
  return SeenBounds.contains(D, C, N);
}

/* Record a drained bounds tuple. Idempotent; call only after a successful drain
 * so a failed attempt stays retryable. */
void __prof_rocm::profRecordDrainedBounds(const void *D, const void *C,
                                          const void *N) {
  SeenBounds.record(D, C, N);
}

#define PROF_MAX_GPU_AGENTS 64

/* Buffer size for HSA agent names and symbol names we read back; both the
 * device arch string and the __llvm_profile_sections symbol are far shorter. */
#define PROF_HSA_NAME_MAX 64

namespace {
struct GpuAgent {
  hsa_agent_t agent;
  char arch[PROF_HSA_NAME_MAX];
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
static hsa_status_t onSymbol(hsa_executable_t, hsa_agent_t,
                             hsa_executable_symbol_t Sym, void *Data) {
  SymbolState *S = (SymbolState *)Data;

  hsa_symbol_kind_t Kind;
  if (!hsaOk(pHsaSymGetInfo(Sym, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &Kind)) ||
      Kind != HSA_SYMBOL_KIND_VARIABLE)
    return HSA_STATUS_SUCCESS;

  uint32_t NameLen = 0;
  if (!hsaOk(pHsaSymGetInfo(Sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH,
                            &NameLen)) ||
      NameLen != sizeof(ProfileSectionsSymbol) - 1)
    return HSA_STATUS_SUCCESS;

  char NameBuf[PROF_HSA_NAME_MAX];
  if (NameLen + 1 > sizeof(NameBuf))
    return HSA_STATUS_SUCCESS;
  if (!hsaOk(pHsaSymGetInfo(Sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME, NameBuf)))
    return HSA_STATUS_SUCCESS;
  NameBuf[NameLen] = '\0';

  if (strcmp(NameBuf, ProfileSectionsSymbol) != 0)
    return HSA_STATUS_SUCCESS;

  uint64_t Addr = 0;
  if (!hsaOk(pHsaSymGetInfo(Sym, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS,
                            &Addr)) ||
      Addr == 0) {
    if (isVerboseMode())
      PROF_NOTE("%s", "failed to read __llvm_profile_sections address\n");
    return HSA_STATUS_SUCCESS;
  }

  S->found++;

  // Read the bounds table first to dedup (and detect empty sections) before
  // the full copy/relocate done by processDeviceOffloadPrf.
  __llvm_profile_gpu_sections Sec;
  if (memcpyDeviceToHost(&Sec, (void *)(uintptr_t)Addr, sizeof(Sec)) != 0) {
    PROF_WARN("%s", "failed to copy device bounds table\n");
    return HSA_STATUS_SUCCESS;
  }
  if (profBoundsAlreadyDrained(Sec.DataStart, Sec.CountersStart,
                               Sec.NamesStart)) {
    if (isVerboseMode())
      PROF_NOTE("%s", "device bounds already drained, skipping\n");
    return HSA_STATUS_SUCCESS;
  }

  size_t DataBytes = (const char *)Sec.DataStop - (const char *)Sec.DataStart;
  size_t CntsBytes =
      (const char *)Sec.CountersStop - (const char *)Sec.CountersStart;
  if (DataBytes == 0 || CntsBytes == 0) {
    // Empty code object: nothing to write. Mark seen so we don't revisit it.
    profRecordDrainedBounds(Sec.DataStart, Sec.CountersStart, Sec.NamesStart);
    return HSA_STATUS_SUCCESS;
  }

  // Name HSA-drained objects in their own ".hsaN" suffix space so they never
  // collide with the host-shadow path's "arch"/"arch.<i>" filenames. The drain
  // latch (HsaDrainCompleted) already prevents re-draining an object, so a
  // plain per-drain counter is enough for uniqueness.
  static int DrainIndex = 0;
  char Target[96];
  snprintf(Target, sizeof(Target), "%s.hsa%d", S->arch, DrainIndex);

  // Record the bounds (and advance the index) only on a successful write so a
  // transient error stays retryable on a later agent or collect call.
  if (processDeviceOffloadPrf((void *)(uintptr_t)Addr, Target, nullptr) == 0) {
    S->drained++;
    DrainIndex++;
    profRecordDrainedBounds(Sec.DataStart, Sec.CountersStart, Sec.NamesStart);
  }

  return HSA_STATUS_SUCCESS;
}

static hsa_status_t collectAgent(hsa_agent_t Agent, void *Data) {
  hsa_device_type_t DevType;
  if (!hsaOk(pHsaAgentGetInfo(Agent, HSA_AGENT_INFO_DEVICE, &DevType)) ||
      DevType != HSA_DEVICE_TYPE_GPU)
    return HSA_STATUS_SUCCESS;

  WalkState *W = (WalkState *)Data;
  if (W->num_agents >= PROF_MAX_GPU_AGENTS)
    return HSA_STATUS_SUCCESS;

  GpuAgent &GA = W->agents[W->num_agents++];
  GA.agent = Agent;
  char Name[PROF_HSA_NAME_MAX];
  __builtin_memset(Name, 0, sizeof(Name));
  pHsaAgentGetInfo(Agent, HSA_AGENT_INFO_NAME, Name);
  size_t N = strnlen(Name, sizeof(GA.arch) - 1);
  __builtin_memcpy(GA.arch, Name, N);
  GA.arch[N] = '\0';
  if (!GA.arch[0])
    strncpy(GA.arch, "amdgpu", sizeof(GA.arch) - 1);

  if (isVerboseMode())
    PROF_NOTE("GPU agent %d: %s\n", W->num_agents - 1, GA.arch);
  return HSA_STATUS_SUCCESS;
}

/* Reentrancy guard and "drained at least once" latch (both acquire/release). */
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

  if (bindHsaRuntime() != 0)
    return 0; /* Runtime unavailable: stay retryable. */

  WalkState W;
  __builtin_memset(&W, 0, sizeof(W));
  hsa_status_t St = pHsaIterateAgents(collectAgent, &W);
  if (!hsaOkOrBreak(St)) {
    PROF_WARN("hsa_iterate_agents failed (0x%x)\n", St);
    return -1;
  }
  if (W.num_agents == 0) {
    if (isVerboseMode())
      PROF_NOTE("%s", "no GPU agents present; nothing to drain (will retry)\n");
    return 0;
  }

  /* query_segment_descriptors ships in every loader-extension version, is more
   * permissive than iterate_executables on ROCm, and yields the loaded
   * (agent, executable) pairs directly. */
  size_t NumSegs = 0;
  St = pQuerySegDescs(nullptr, &NumSegs);
  if (!hsaOk(St)) {
    PROF_WARN("query_segment_descriptors(count) failed (0x%x)\n", St);
    return -1;
  }
  if (NumSegs == 0) {
    if (isVerboseMode())
      PROF_NOTE("%s", "no loaded segments; nothing to drain (will retry)\n");
    return 0;
  }

  hsa_ven_amd_loader_segment_descriptor_t *Segs =
      (hsa_ven_amd_loader_segment_descriptor_t *)calloc(NumSegs, sizeof(*Segs));
  if (!Segs) {
    PROF_ERR("%s\n", "failed to allocate segment descriptor array");
    return -1;
  }
  UniqueFree SegsOwner(Segs);

  St = pQuerySegDescs(Segs, &NumSegs);
  if (!hsaOk(St)) {
    PROF_WARN("query_segment_descriptors(fetch) failed (0x%x)\n", St);
    return -1;
  }

  if (isVerboseMode())
    PROF_NOTE("query_segment_descriptors: %zu segments\n", NumSegs);

  // Walk each unique (agent, executable) pair once.
  struct SeenPair {
    uint64_t agent;
    uint64_t exec;
  };
  enum { kSeenPairsInitCap = 64 };
  SeenPair *Seen = nullptr;
  int NumPairs = 0;
  int CapPairs = 0;
  int IterFailures = 0;

  for (size_t i = 0; i < NumSegs; ++i) {
    if (Segs[i].executable.handle == 0 || Segs[i].agent.handle == 0)
      continue;

    bool AlreadySeen = false;
    for (int j = 0; j < NumPairs; ++j)
      if (Seen[j].agent == Segs[i].agent.handle &&
          Seen[j].exec == Segs[i].executable.handle) {
        AlreadySeen = true;
        break;
      }
    if (AlreadySeen)
      continue;
    if (growArray((void **)&Seen, &CapPairs, NumPairs + 1, kSeenPairsInitCap,
                  sizeof(*Seen)) == 0) {
      Seen[NumPairs].agent = Segs[i].agent.handle;
      Seen[NumPairs].exec = Segs[i].executable.handle;
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
    hsa_status_t IterSt =
        pHsaExecIterAgentSyms(Segs[i].executable, Segs[i].agent, onSymbol, &S);
    if (!hsaOkOrBreak(IterSt)) {
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

  free(Seen);

  /* Latch only when we actually drained data. A "found nothing new" walk is
   * deliberately not latched: an early collect can precede any kernel launch,
   * and latching it would suppress the real exit-time drain. No-op walks are
   * cheap to repeat. */
  if (W.total_drained > 0)
    __atomic_store_n(&HsaDrainCompleted, 1, __ATOMIC_RELEASE);
  return (IterFailures > 0) ? -1 : 0;
}

/* Fork-safety: deliberately no library constructor calling hsa_init(). */

#endif /* defined(__linux__) */
