//===-- asan_globals.cc ---------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Handle globals.
//===----------------------------------------------------------------------===//

#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_poisoning.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_stats.h"
#include "asan_suppressions.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

namespace __asan {

typedef __asan_global Global;

struct ListOfGlobals {
  const Global *g;
  ListOfGlobals *next;
};

static BlockingMutex mu_for_globals(LINKER_INITIALIZED);
static LowLevelAllocator allocator_for_globals;
static ListOfGlobals *list_of_all_globals;

static const int kDynamicInitGlobalsInitialCapacity = 512;
struct DynInitGlobal {
  Global g;
  bool initialized;
};
typedef InternalMmapVector<DynInitGlobal> VectorOfGlobals;
// Lazy-initialized and never deleted.
static VectorOfGlobals *dynamic_init_globals;

// We want to remember where a certain range of globals was registered.
struct GlobalRegistrationSite {
  u32 stack_id;
  Global *g_first, *g_last;
};
typedef InternalMmapVector<GlobalRegistrationSite> GlobalRegistrationSiteVector;
static GlobalRegistrationSiteVector *global_registration_site_vector;

ALWAYS_INLINE void PoisonShadowForGlobal(const Global *g, u8 value) {
  FastPoisonShadow(g->beg, g->size_with_redzone, value);
}

ALWAYS_INLINE void PoisonRedZones(const Global &g) {
  uptr aligned_size = RoundUpTo(g.size, SHADOW_GRANULARITY);
  FastPoisonShadow(g.beg + aligned_size, g.size_with_redzone - aligned_size,
                   kAsanGlobalRedzoneMagic);
  if (g.size != aligned_size) {
    FastPoisonShadowPartialRightRedzone(
        g.beg + RoundDownTo(g.size, SHADOW_GRANULARITY),
        g.size % SHADOW_GRANULARITY,
        SHADOW_GRANULARITY,
        kAsanGlobalRedzoneMagic);
  }
}

const uptr kMinimalDistanceFromAnotherGlobal = 64;

static bool IsAddressNearGlobal(uptr addr, const __asan_global &g) {
  if (addr <= g.beg - kMinimalDistanceFromAnotherGlobal) return false;
  if (addr >= g.beg + g.size_with_redzone) return false;
  return true;
}

static void ReportGlobal(const Global &g, const char *prefix) {
  Report("%s Global[%p]: beg=%p size=%zu/%zu name=%s module=%s dyn_init=%zu\n",
         prefix, &g, (void *)g.beg, g.size, g.size_with_redzone, g.name,
         g.module_name, g.has_dynamic_init);
  if (g.location) {
    Report("  location (%p): name=%s[%p], %d %d\n", g.location,
           g.location->filename, g.location->filename, g.location->line_no,
           g.location->column_no);
  }
}

static u32 FindRegistrationSite(const Global *g) {
  mu_for_globals.CheckLocked();
  CHECK(global_registration_site_vector);
  for (uptr i = 0, n = global_registration_site_vector->size(); i < n; i++) {
    GlobalRegistrationSite &grs = (*global_registration_site_vector)[i];
    if (g >= grs.g_first && g <= grs.g_last)
      return grs.stack_id;
  }
  return 0;
}

int GetGlobalsForAddress(uptr addr, Global *globals, u32 *reg_sites,
                         int max_globals) {
  if (!flags()->report_globals) return 0;
  BlockingMutexLock lock(&mu_for_globals);
  int res = 0;
  for (ListOfGlobals *l = list_of_all_globals; l; l = l->next) {
    const Global &g = *l->g;
    if (flags()->report_globals >= 2)
      ReportGlobal(g, "Search");
    if (IsAddressNearGlobal(addr, g)) {
      globals[res] = g;
      if (reg_sites)
        reg_sites[res] = FindRegistrationSite(&g);
      res++;
      if (res == max_globals) break;
    }
  }
  return res;
}

bool GetInfoForAddressIfGlobal(uptr addr, AddressDescription *descr) {
  Global g = {};
  if (GetGlobalsForAddress(addr, &g, nullptr, 1)) {
    internal_strncpy(descr->name, g.name, descr->name_size);
    descr->region_address = g.beg;
    descr->region_size = g.size;
    descr->region_kind = "global";
    return true;
  }
  return false;
}

// Register a global variable.
// This function may be called more than once for every global
// so we store the globals in a map.
static void RegisterGlobal(const Global *g) {
  CHECK(asan_inited);
  if (flags()->report_globals >= 2)
    ReportGlobal(*g, "Added");
  CHECK(flags()->report_globals);
  CHECK(AddrIsInMem(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->size_with_redzone));
  if (flags()->detect_odr_violation) {
    // Try detecting ODR (One Definition Rule) violation, i.e. the situation
    // where two globals with the same name are defined in different modules.
    if (__asan_region_is_poisoned(g->beg, g->size_with_redzone)) {
      // This check may not be enough: if the first global is much larger
      // the entire redzone of the second global may be within the first global.
      for (ListOfGlobals *l = list_of_all_globals; l; l = l->next) {
        if (g->beg == l->g->beg &&
            (flags()->detect_odr_violation >= 2 || g->size != l->g->size) &&
            !IsODRViolationSuppressed(g->name))
          ReportODRViolation(g, FindRegistrationSite(g),
                             l->g, FindRegistrationSite(l->g));
      }
    }
  }
  if (CanPoisonMemory())
    PoisonRedZones(*g);
  ListOfGlobals *l = new(allocator_for_globals) ListOfGlobals;
  l->g = g;
  l->next = list_of_all_globals;
  list_of_all_globals = l;
  if (g->has_dynamic_init) {
    if (!dynamic_init_globals) {
      dynamic_init_globals = new(allocator_for_globals)
          VectorOfGlobals(kDynamicInitGlobalsInitialCapacity);
    }
    DynInitGlobal dyn_global = { *g, false };
    dynamic_init_globals->push_back(dyn_global);
  }
}

static void UnregisterGlobal(const Global *g) {
  CHECK(asan_inited);
  if (flags()->report_globals >= 2)
    ReportGlobal(*g, "Removed");
  CHECK(flags()->report_globals);
  CHECK(AddrIsInMem(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->size_with_redzone));
  if (CanPoisonMemory())
    PoisonShadowForGlobal(g, 0);
  // We unpoison the shadow memory for the global but we do not remove it from
  // the list because that would require O(n^2) time with the current list
  // implementation. It might not be worth doing anyway.
}

void StopInitOrderChecking() {
  BlockingMutexLock lock(&mu_for_globals);
  if (!flags()->check_initialization_order || !dynamic_init_globals)
    return;
  flags()->check_initialization_order = false;
  for (uptr i = 0, n = dynamic_init_globals->size(); i < n; ++i) {
    DynInitGlobal &dyn_g = (*dynamic_init_globals)[i];
    const Global *g = &dyn_g.g;
    // Unpoison the whole global.
    PoisonShadowForGlobal(g, 0);
    // Poison redzones back.
    PoisonRedZones(*g);
  }
}

} // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

// Register an array of globals.
void __asan_register_globals(__asan_global *globals, uptr n) {
  if (!flags()->report_globals) return;
  GET_STACK_TRACE_MALLOC;
  u32 stack_id = StackDepotPut(stack);
  BlockingMutexLock lock(&mu_for_globals);
  if (!global_registration_site_vector)
    global_registration_site_vector =
        new(allocator_for_globals) GlobalRegistrationSiteVector(128);
  GlobalRegistrationSite site = {stack_id, &globals[0], &globals[n - 1]};
  global_registration_site_vector->push_back(site);
  if (flags()->report_globals >= 2) {
    PRINT_CURRENT_STACK();
    Printf("=== ID %d; %p %p\n", stack_id, &globals[0], &globals[n - 1]);
  }
  for (uptr i = 0; i < n; i++) {
    RegisterGlobal(&globals[i]);
  }
}

// Unregister an array of globals.
// We must do this when a shared objects gets dlclosed.
void __asan_unregister_globals(__asan_global *globals, uptr n) {
  if (!flags()->report_globals) return;
  BlockingMutexLock lock(&mu_for_globals);
  for (uptr i = 0; i < n; i++) {
    UnregisterGlobal(&globals[i]);
  }
}

// This method runs immediately prior to dynamic initialization in each TU,
// when all dynamically initialized globals are unpoisoned.  This method
// poisons all global variables not defined in this TU, so that a dynamic
// initializer can only touch global variables in the same TU.
void __asan_before_dynamic_init(const char *module_name) {
  if (!flags()->check_initialization_order ||
      !CanPoisonMemory())
    return;
  bool strict_init_order = flags()->strict_init_order;
  CHECK(dynamic_init_globals);
  CHECK(module_name);
  CHECK(asan_inited);
  BlockingMutexLock lock(&mu_for_globals);
  if (flags()->report_globals >= 3)
    Printf("DynInitPoison module: %s\n", module_name);
  for (uptr i = 0, n = dynamic_init_globals->size(); i < n; ++i) {
    DynInitGlobal &dyn_g = (*dynamic_init_globals)[i];
    const Global *g = &dyn_g.g;
    if (dyn_g.initialized)
      continue;
    if (g->module_name != module_name)
      PoisonShadowForGlobal(g, kAsanInitializationOrderMagic);
    else if (!strict_init_order)
      dyn_g.initialized = true;
  }
}

// This method runs immediately after dynamic initialization in each TU, when
// all dynamically initialized globals except for those defined in the current
// TU are poisoned.  It simply unpoisons all dynamically initialized globals.
void __asan_after_dynamic_init() {
  if (!flags()->check_initialization_order ||
      !CanPoisonMemory())
    return;
  CHECK(asan_inited);
  BlockingMutexLock lock(&mu_for_globals);
  // FIXME: Optionally report that we're unpoisoning globals from a module.
  for (uptr i = 0, n = dynamic_init_globals->size(); i < n; ++i) {
    DynInitGlobal &dyn_g = (*dynamic_init_globals)[i];
    const Global *g = &dyn_g.g;
    if (!dyn_g.initialized) {
      // Unpoison the whole global.
      PoisonShadowForGlobal(g, 0);
      // Poison redzones back.
      PoisonRedZones(*g);
    }
  }
}
