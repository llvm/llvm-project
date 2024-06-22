#include "InstrProfilingTLS.h"
#include "InstrProfiling.h"

struct texit_fn_node module_node COMPILER_RT_VISIBILITY;

// We act as a shim between the profile_threadlocal sharedlib
// and the profile static lib.  We need to the tell the static lib
// to add all of the counters up on main thread exit, but the
// shared lib is the one who knows how to do that and whether its
// already been done.
//
// In the constructor we pass flush_main_thread_counters from the
// sharedlib to the non-tls statlib's on_main_thread_exit fnptr.
extern void flush_main_thread_counters(void);
extern void (*on_main_thread_exit)(void);

__attribute__((constructor)) COMPILER_RT_VISIBILITY void
__llvm_profile_tls_register_thread_exit_handler(void) {
  module_node.prev = NULL;
  module_node.next = NULL;
  module_node.fn = __llvm_profile_tls_counters_finalize;
  register_tls_prfcnts_module_thread_exit_handler(&module_node);
  if (!on_main_thread_exit) {
    on_main_thread_exit = flush_main_thread_counters;
  }
}

// TODO: Add destructor
// (But not yet, I'm scared)
