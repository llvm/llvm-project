#include "InstrProfiling.h"
#include "InstrProfilingTLS.h"
#include <stdlib.h>

// Maintain a linked list of handlers to run on thread exit.
// This is broken out into a dylib so that the registry is truly global across
// dlopen et. al.
//
// Each module has a statically allocated node that gets linked into the
// registry on the constructor and that gets linked out of the registry on
// destroy.
//
// This node is defined in the static portion of the tls counts extension.

struct texit_fn_registry texit_registry;

static void lock_texit_registry(void) {
  int expected = 0;
  while (!__atomic_compare_exchange_n(&texit_registry.texit_mtx, &expected, 1,
                                      0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
    expected = 0;
  }
}

static void unlock_texit_registry(void) {
  __atomic_store_n(&texit_registry.texit_mtx, 0, __ATOMIC_RELEASE);
}

static void wlock_texit_registry(void) { lock_texit_registry(); }

static void wunlock_texit_registry(void) { unlock_texit_registry(); }

static void rlock_texit_registry(void) { lock_texit_registry(); }

static void runlock_texit_registry(void) { unlock_texit_registry(); }

static inline texit_fn_node *take_nodep(texit_fn_node **nodepp) {
  texit_fn_node *nodep = *nodepp;
  *nodepp = NULL;
  return nodep;
}

static inline texit_fn_node *replace_nodep(texit_fn_node **nodepp,
                                           texit_fn_node *new_nodep) {
  texit_fn_node *nodep = *nodepp;
  *nodepp = new_nodep;
  return nodep;
}

void flush_main_thread_counters(void) {
  static int flushed = 0;
  if (!flushed) {
    run_thread_exit_handlers();
    flushed = 1;
  }
}

__attribute__((constructor)) static void __initialize_tls_exit_registry() {
  register_profile_intercepts();
  texit_registry.texit_mtx = 0;
  texit_registry.head.prev = NULL;
  texit_registry.head.fn = NULL;
  texit_registry.head.next = &texit_registry.tail;
  texit_registry.tail.prev = &texit_registry.head;
  texit_registry.tail.fn = NULL;
  texit_registry.tail.next = NULL;
}

// Should run from module constructor
void register_tls_prfcnts_module_thread_exit_handler(texit_fn_node *new_nodep) {
  wlock_texit_registry();
  texit_fn_node *prev = replace_nodep(&texit_registry.tail.prev, new_nodep);
  texit_fn_node *next = replace_nodep(&prev->next, new_nodep);
  new_nodep->next = next;
  new_nodep->prev = prev;
  wunlock_texit_registry();
}

// Should run from module destructor
// Also, this destructor/constructor pair should be outermost.  At least outside
// of the regular llvm_profile stuff.
void unregister_tls_prfcnts_module_thread_exit_handler(
    texit_fn_node *old_nodep) {
  wlock_texit_registry();
  texit_fn_node *prev = take_nodep(&old_nodep->prev);
  texit_fn_node *next = take_nodep(&old_nodep->next);
  prev->next = next;
  next->prev = prev;
  wunlock_texit_registry();
}

void run_thread_exit_handlers(void) {
  rlock_texit_registry();
  for (texit_fn_node *node = texit_registry.head.next;
       node != &texit_registry.tail; node = node->next) {
    if (node->fn != NULL)
      node->fn();
  }
  runlock_texit_registry();
}
