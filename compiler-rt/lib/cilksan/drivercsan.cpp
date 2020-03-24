// #include <cilk/hyperobject_base.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <execinfo.h>
// #include <internal/abi.h>
#include <unistd.h>
#include <map>
#include <unordered_map>
#include <sys/mman.h>

#if CILKSAN_DYNAMIC
#include <dlfcn.h>
#endif  // CILKSAN_DYNAMIC

#include "cilksan_internal.h"
#include "debug_util.h"
#include "stack.h"

#define CILKSAN_API extern "C" __attribute__((visibility("default")))
#define CALLERPC ((uintptr_t)__builtin_return_address(0))

// global var: FILE io used to print error messages
FILE *err_io;

extern CilkSanImpl_t CilkSanImpl;

// declared in cilksan.cpp
extern uintptr_t stack_low_addr;
extern uintptr_t stack_high_addr;

// Defined in print_addr.cpp
// extern void read_proc_maps();
extern void delete_proc_maps();
extern void print_addr(FILE *f, void *a);
// declared in cilksan; for debugging only
#if CILKSAN_DEBUG
extern enum EventType_t last_event;
#endif

// Defined in print_addr.cpp
extern uintptr_t *call_pc;
extern uintptr_t *spawn_pc;
extern uintptr_t *loop_pc;
extern uintptr_t *load_pc;
extern uintptr_t *store_pc;
extern uintptr_t *alloca_pc;
extern uintptr_t *allocfn_pc;
extern allocfn_prop_t *allocfn_prop;
extern uintptr_t *free_pc;
static csi_id_t total_call = 0;
static csi_id_t total_spawn = 0;
static csi_id_t total_loop = 0;
static csi_id_t total_load = 0;
static csi_id_t total_store = 0;
static csi_id_t total_alloca = 0;
static csi_id_t total_allocfn = 0;
static csi_id_t total_free = 0;

static bool TOOL_INITIALIZED = false;

// When either is set to false, no errors are output
static bool instrumentation = false;
// needs to be reentrant due to reducer operations; 0 means checking
static int checking_disabled = 0;

static inline void enable_instrumentation() {
  DBG_TRACE(DEBUG_BASIC, "Enable instrumentation.\n");
  instrumentation = true;
}

static inline void disable_instrumentation() {
  DBG_TRACE(DEBUG_BASIC, "Disable instrumentation.\n");
  instrumentation = false;
}

static inline void enable_checking() {
  checking_disabled--;
  DBG_TRACE(DEBUG_BASIC, "%d: Enable checking.\n", checking_disabled);
  cilksan_assert(checking_disabled >= 0);
}

static inline void disable_checking() {
  cilksan_assert(checking_disabled >= 0);
  checking_disabled++;
  DBG_TRACE(DEBUG_BASIC, "%d: Disable checking.\n", checking_disabled);
}

// RAII object to disable checking in tool routines.
struct CheckingRAII {
  CheckingRAII() {
    disable_checking();
  }
  ~CheckingRAII() {
    enable_checking();
  }
};

// outside world (including runtime).
// Non-inlined version for user code to use
CILKSAN_API void __cilksan_enable_checking() {
  checking_disabled--;
  cilksan_assert(checking_disabled >= 0);
  DBG_TRACE(DEBUG_BASIC, "External enable checking (%d).\n", checking_disabled);
}

// Non-inlined version for user code to use
CILKSAN_API void __cilksan_disable_checking() {
  cilksan_assert(checking_disabled >= 0);
  checking_disabled++;
  DBG_TRACE(DEBUG_BASIC, "External disable checking (%d).\n", checking_disabled);
}

// Non-inlined callback for user code to check if checking is enabled.
CILKSAN_API bool __cilksan_is_checking_enabled() {
  return (checking_disabled == 0);
}

static inline bool should_check() {
  return (instrumentation && (checking_disabled == 0));
}

Stack_t<uint8_t> parallel_execution;

Stack_t<std::pair<csi_id_t, uint64_t>> suppressions;
Stack_t<uint64_t> suppression_counts;

CILKSAN_API void __csan_set_suppression_flag(uint64_t val, csi_id_t id) {
  DBG_TRACE(DEBUG_CALLBACK, "__csan_set_suppression_flag(%ld, %ld)\n",
            val, id);
  suppressions.push();
  *suppressions.head() = std::make_pair(id, val);
}

CILKSAN_API void __csan_get_suppression_flag(uint64_t *ptr, csi_id_t id,
                                             unsigned idx) {
  DBG_TRACE(DEBUG_CALLBACK, "__csan_get_suppression_flag(%x, %d, %d)\n",
            ptr, id, idx);
  // We presume that __csan_get_suppression_flag runs early in the function, so
  // if instrumentation is disabled, it's disabled for the whole function.
  if (!should_check()) {
    *ptr = /*NoModRef*/0;
    return;
  }

  unsigned suppression_count = *suppression_counts.head();
  if (idx >= suppression_count) {
    DBG_TRACE(DEBUG_CALLBACK, "  No suppression found: idx %d >= count %d\n",
              idx, suppression_count);
    // The stack doesn't have suppressions for us, so assume the worst.
    *ptr = /*ModRef*/3;
    return;
  }

  std::pair<csi_id_t, unsigned> suppression = *suppressions.ancestor(idx);
  if (suppression.first == id) {
    DBG_TRACE(DEBUG_CALLBACK, "  Found suppression: %d\n",
              suppression.second);
    *ptr = suppression.second;
  } else {
    DBG_TRACE(DEBUG_CALLBACK, "  No suppression found\n");
    // The stack doesn't have suppressions for us, so assume the worst.
    *ptr = /*ModRef*/3;
  }
}

// called upon process exit
static void csan_destroy(void) {
  disable_instrumentation();
  disable_checking();
  CilkSanImpl.deinit();
  fflush(stdout);
  delete_proc_maps();
  if (call_pc) {
    free(call_pc);
    call_pc = nullptr;
  }
  if (spawn_pc) {
    free(spawn_pc);
    spawn_pc = nullptr;
  }
  if (loop_pc) {
    free(loop_pc);
    loop_pc = nullptr;
  }
  if (load_pc) {
    free(load_pc);
    load_pc = nullptr;
  }
  if (store_pc) {
    free(store_pc);
    store_pc = nullptr;
  }
  if (alloca_pc) {
    free(alloca_pc);
    alloca_pc = nullptr;
  }
  if (allocfn_pc) {
    free(allocfn_pc);
    allocfn_pc = nullptr;
  }
  if (allocfn_prop) {
    free(allocfn_prop);
    allocfn_prop = nullptr;
  }
  if (free_pc) {
    free(free_pc);
    free_pc = nullptr;
  }
}

CilkSanImpl_t::~CilkSanImpl_t() {
  csan_destroy();
}

static void init_internal() {
  // read_proc_maps();
  if (ERROR_FILE) {
    FILE *tmp = fopen(ERROR_FILE, "w+");
    if (tmp) err_io = tmp;
  }
  if (err_io == NULL) err_io = stderr;

  // Force the number of Cilk workers to be 1.
  char *e = getenv("CILK_NWORKERS");
  if (!e || 0 != strcmp(e, "1")) {
    // fprintf(err_io, "Setting CILK_NWORKERS to be 1\n");
    if (setenv("CILK_NWORKERS", "1", 1)) {
      fprintf(err_io, "Error setting CILK_NWORKERS to be 1\n");
      exit(1);
    }
  }

  // Force reductions.
  // XXX: Does not work with SP+ algorithm, but works with ordinary
  // SP bags.
  e = getenv("CILK_FORCE_REDUCE");
  if (!e || 0 != strcmp(e, "1")) {
    // fprintf(err_io, "Setting CILK_FORCE_REDUCE to be 1\n");
    if (setenv("CILK_FORCE_REDUCE", "1", 1)) {
      fprintf(err_io, "Error setting CILK_FORCE_REDUCE to be 1\n");
      exit(1);
    }
  }
}

CILKSAN_API void __csi_init() {
  // This method should only be called once.
  cilksan_assert(!TOOL_INITIALIZED && "__csi_init() called multiple times.");

  // We use the automatic deallocation of the CilkSanImpl top-level tool object
  // to shutdown and cleanup the tool at program termination.
  // atexit(csan_destroy);

  init_internal();
  // moved this later when we enter the first Cilk frame
  // cilksan_init();
  // enable_instrumentation();
  TOOL_INITIALIZED = true;
  // fprintf(err_io, "tsan_init called.\n");
}

// Helper function to grow a map from CSI ID to program counter (PC).
static void grow_pc_table(uintptr_t *&table, csi_id_t &table_cap,
                          csi_id_t extra_cap) {
  csi_id_t new_cap = table_cap + extra_cap;
  table = (uintptr_t *)realloc(table, new_cap * sizeof(uintptr_t));
  for (csi_id_t i = table_cap; i < new_cap; ++i)
    table[i] = (uintptr_t)nullptr;
  table_cap = new_cap;
}

CILKSAN_API
void __csan_unit_init(const char * const file_name,
                      const csan_instrumentation_counts_t counts) {
  CheckingRAII nocheck;
  // Grow the tables mapping CSI ID's to PC values.
  if (counts.num_call)
    grow_pc_table(call_pc, total_call, counts.num_call);
  if (counts.num_detach)
    grow_pc_table(spawn_pc, total_spawn, counts.num_detach);
  if (counts.num_loop)
    grow_pc_table(loop_pc, total_loop, counts.num_loop);
  if (counts.num_load)
    grow_pc_table(load_pc, total_load, counts.num_load);
  if (counts.num_store)
    grow_pc_table(store_pc, total_store, counts.num_store);
  if (counts.num_alloca)
    grow_pc_table(alloca_pc, total_alloca, counts.num_alloca);
  if (counts.num_allocfn) {
    csi_id_t new_cap = total_allocfn + counts.num_allocfn;
    allocfn_prop = (allocfn_prop_t *)realloc(allocfn_prop,
                                             new_cap * sizeof(allocfn_prop_t));
    for (csi_id_t i = total_allocfn; i < new_cap; ++i)
      allocfn_prop[i].allocfn_ty = uint8_t(-1);
    grow_pc_table(allocfn_pc, total_allocfn, counts.num_allocfn);
  }
  if (counts.num_free)
    grow_pc_table(free_pc, total_free, counts.num_free);
}

// invoked whenever a function enters; no need for this
CILKSAN_API void __csan_func_entry(const csi_id_t func_id,
                                   const void *bp, const void *sp,
                                   const func_prop_t prop) {
  { // Handle tool initialization as a special case.
    CheckingRAII nocheck_init;
    static bool first_call = true;
    if (first_call) {
      CilkSanImpl.init();
      enable_instrumentation();
      // Note that we start executing the program in series.
      parallel_execution.push();
      *parallel_execution.head() = 0;
      first_call = false;
    }
  }

  // fprintf(stderr, "__csan_func_entry: bp = %p, sp = %p\n",
  //         (uintptr_t)bp, (uintptr_t)sp);
  if (stack_high_addr < (uintptr_t)bp)
    stack_high_addr = (uintptr_t)bp;
  if (stack_low_addr > (uintptr_t)sp)
    stack_low_addr = (uintptr_t)sp;

  if (!should_check())
    return;

  CheckingRAII nocheck;
  WHEN_CILKSAN_DEBUG({
      const csan_source_loc_t *srcloc = __csan_get_func_source_loc(func_id);
      DBG_TRACE(DEBUG_CALLBACK, "__csan_func_entry(%d) at %s (%s:%d)\n",
                func_id,
                srcloc->name, srcloc->filename,
                srcloc->line_number);
    });
  cilksan_assert(TOOL_INITIALIZED);

  // Propagate the parallel-execution state to the child.
  uint8_t current_pe = *parallel_execution.head();
  // First we push the pe value on function entry.
  parallel_execution.push();
  *parallel_execution.head() = current_pe;
  // We push a second copy to update aggressively on detaches.
  parallel_execution.push();
  *parallel_execution.head() = current_pe;

  CilkSanImpl.push_stack_frame((uintptr_t)bp, (uintptr_t)sp);

  // if (!prop.may_spawn)
  //   // Ignore entry calls into non-Cilk functions.
  //   return;

  // Update the tool for entering a Cilk function.
  CilkSanImpl.do_enter_begin(prop.num_sync_reg);
  CilkSanImpl.do_enter_end((uintptr_t)sp);
  enable_instrumentation();
}

CILKSAN_API void __csan_func_exit(const csi_id_t func_exit_id,
                                  const csi_id_t func_id,
                                  const func_exit_prop_t prop) {
  if (!should_check())
    return;

  CheckingRAII nocheck;
  cilksan_assert(TOOL_INITIALIZED);
#if CILKSAN_DEBUG
  const csan_source_loc_t *srcloc = __csan_get_func_exit_source_loc(func_exit_id);
#endif
  DBG_TRACE(DEBUG_CALLBACK, "__csan_func_exit(%ld, %ld) at %s (%s:%d)\n",
            func_exit_id, func_id,
            srcloc->name, srcloc->filename,
            srcloc->line_number);

  // if (prop.may_spawn) {
    // Update the tool for leaving a Cilk function.
    //
    // NOTE: Technically the sync region that would synchronize any orphaned
    // child tasks is not well defined.  This case should never arise in Cilk
    // programs.
    CilkSanImpl.do_leave_begin(0);
    CilkSanImpl.do_leave_end();
  // }

  // Pop both local copies of the parallel-execution state.
  parallel_execution.pop();
  parallel_execution.pop();

  CilkSanImpl.pop_stack_frame();

  // XXX Let's focus on Cilk function for now; maybe put it back later
  // cilksan_do_function_exit();
}

CILKSAN_API void __csan_before_loop(const csi_id_t loop_id,
                                    const int64_t trip_count,
                                    const loop_prop_t prop) {
  if (!should_check())
    return;

  if (!prop.is_tapir_loop)
    return;

  CheckingRAII nocheck;
  DBG_TRACE(DEBUG_CALLBACK, "__csan_before_loop(%ld)\n", loop_id);

  // Record the address of this parallel loop.
  if (__builtin_expect(!loop_pc[loop_id], false))
    loop_pc[loop_id] = CALLERPC;

  // Push the parallel loop onto the call stack.
  CilkSanImpl.record_call(loop_id, LOOP);

  // Propagate the parallel-execution state to the child.
  uint8_t current_pe = *parallel_execution.head();
  // First push the pe value on function entry.
  parallel_execution.push();
  *parallel_execution.head() = current_pe;
  // Push an extra copy to the head, to be updated aggressively due to detaches.
  parallel_execution.push();
  *parallel_execution.head() = current_pe;

  CilkSanImpl.do_loop_begin();
}

CILKSAN_API void __csan_after_loop(const csi_id_t loop_id,
                                   const unsigned sync_reg,
                                   const loop_prop_t prop) {
  if (!should_check())
    return;

  if (!prop.is_tapir_loop)
    return;

  CheckingRAII nocheck;
  DBG_TRACE(DEBUG_CALLBACK, "__csan_after_loop(%ld)\n", loop_id);

  CilkSanImpl.do_loop_end(sync_reg);

  // Pop the parallel-execution state.
  parallel_execution.pop();
  parallel_execution.pop();

  // Pop the call off of the call stack.
  CilkSanImpl.record_call_return(loop_id, LOOP);
}

CILKSAN_API void __csan_before_call(const csi_id_t call_id,
                                    const csi_id_t func_id,
                                    unsigned suppression_count,
                                    const call_prop_t prop) {
  if (!should_check())
    return;

  CheckingRAII nocheck;
  DBG_TRACE(DEBUG_CALLBACK, "__csan_before_call(%ld, %ld)\n",
            call_id, func_id);

  // Record the address of this call site.
  if (__builtin_expect(!call_pc[call_id], false))
    call_pc[call_id] = CALLERPC;

  // Push the suppression count onto the stack.
  suppression_counts.push();
  *suppression_counts.head() = suppression_count;
  // fprintf(stderr, "suppression count %d\n", suppression_count);

  // Push the call onto the call stack.
  CilkSanImpl.record_call(call_id, CALL);
}

CILKSAN_API void __csan_after_call(const csi_id_t call_id,
                                   const csi_id_t func_id,
                                   unsigned suppression_count,
                                   const call_prop_t prop) {
  if (!should_check())
    return;

  CheckingRAII nocheck;
  DBG_TRACE(DEBUG_CALLBACK, "__csan_after_call(%ld, %ld)\n",
            call_id, func_id);

  // Pop any suppressions.
  for (unsigned i = 0; i < suppression_count; ++i)
    suppressions.pop();
  suppression_counts.pop();

  // Pop the call off of the call stack.
  CilkSanImpl.record_call_return(call_id, CALL);
}

CILKSAN_API void __csan_detach(const csi_id_t detach_id,
                               const unsigned sync_reg) {
  if (!should_check())
    return;

  CheckingRAII nocheck;
  DBG_TRACE(DEBUG_CALLBACK, "__csan_detach(%ld)\n",
            detach_id);
  cilksan_assert(last_event == NONE);
  WHEN_CILKSAN_DEBUG(last_event = SPAWN_PREPARE);
  WHEN_CILKSAN_DEBUG(last_event = NONE);

  // Record the address of this detach.
  if (__builtin_expect(!spawn_pc[detach_id], false))
    spawn_pc[detach_id] = CALLERPC;

  // Update the parallel-execution state to reflect this detach.  Essentially,
  // this notes the change of peer sets.
  *parallel_execution.head() = 1;

  if (!CilkSanImpl.handle_loop())
    // Push the detach onto the call stack.
    CilkSanImpl.record_call(detach_id, SPAWN);
}

CILKSAN_API void __csan_task(const csi_id_t task_id, const csi_id_t detach_id,
                             const void *bp, const void *sp,
                             const task_prop_t prop) {
  if (!should_check())
    return;
  // fprintf(stderr, "__csan_task: bp = %p, sp = %p\n",
  //         (uintptr_t)bp, (uintptr_t)sp);

  if (stack_low_addr > (uintptr_t)sp)
    stack_low_addr = (uintptr_t)sp;

  CheckingRAII nocheck;
  DBG_TRACE(DEBUG_CALLBACK, "__csan_task(%ld, %ld, %d)\n",
            task_id, detach_id, prop.is_tapir_loop_body);
  WHEN_CILKSAN_DEBUG(last_event = NONE);

  CilkSanImpl.push_stack_frame((uintptr_t)bp, (uintptr_t)sp);

  if (prop.is_tapir_loop_body && CilkSanImpl.handle_loop()) {
    CilkSanImpl.do_loop_iteration_begin((uintptr_t)sp, prop.num_sync_reg);
    return;
  }

  // Propagate the parallel-execution state to the child.
  uint8_t current_pe = *parallel_execution.head();
  // Push the pe value on function entry.
  parallel_execution.push();
  *parallel_execution.head() = current_pe;
  // Push a second copy to update aggressively on detaches.
  parallel_execution.push();
  *parallel_execution.head() = current_pe;

  // Update tool for entering detach-helper function and performing detach.
  CilkSanImpl.do_enter_helper_begin(prop.num_sync_reg);
  CilkSanImpl.do_enter_end((uintptr_t)sp);
  CilkSanImpl.do_detach_begin();
  CilkSanImpl.do_detach_end();
}

CILKSAN_API void __csan_task_exit(const csi_id_t task_exit_id,
                                  const csi_id_t task_id,
                                  const csi_id_t detach_id,
                                  const unsigned sync_reg,
                                  const task_exit_prop_t prop) {
  if (!should_check())
    return;

  CheckingRAII nocheck;
  DBG_TRACE(DEBUG_CALLBACK, "__csan_task_exit(%ld, %ld, %ld, %d)\n",
            task_exit_id, task_id, detach_id, prop.is_tapir_loop_body);

  if (prop.is_tapir_loop_body && CilkSanImpl.handle_loop()) {
    // Update tool for leaving the parallel iteration.
    CilkSanImpl.do_loop_iteration_end();

    // The parallel-execution state will be popped when the loop terminates.
  } else {
    // Update tool for leaving a detach-helper function.
    CilkSanImpl.do_leave_begin(sync_reg);
    CilkSanImpl.do_leave_end();

    // Pop the parallel-execution state.
    parallel_execution.pop();
    parallel_execution.pop();
  }

  CilkSanImpl.pop_stack_frame();
}

CILKSAN_API void __csan_detach_continue(const csi_id_t detach_continue_id,
                                        const csi_id_t detach_id) {
  if (!should_check())
    return;
  CheckingRAII nocheck;
  DBG_TRACE(DEBUG_CALLBACK, "__csan_detach_continue(%ld)\n",
            detach_id);

  if (!CilkSanImpl.handle_loop())
    CilkSanImpl.record_call_return(detach_id, SPAWN);

  WHEN_CILKSAN_DEBUG({
      if (last_event == LEAVE_FRAME_OR_HELPER)
        CilkSanImpl.do_leave_end();
    });
  WHEN_CILKSAN_DEBUG(last_event = NONE);
}

CILKSAN_API void __csan_sync(csi_id_t sync_id, const unsigned sync_reg) {
  if (!should_check())
    return;
  CheckingRAII nocheck;
  cilksan_assert(TOOL_INITIALIZED);

  // Because this is a serial tool, we can safely perform all operations related
  // to a sync.
  CilkSanImpl.do_sync_begin();
  CilkSanImpl.do_sync_end(sync_reg);

  // Restore the parallel-execution state to that of the function/task entry.
  if (CilkSanImpl.is_local_synced())
    *parallel_execution.head() = *parallel_execution.ancestor(1);
}

// Assuming __csan_load/store is inlined, the stack should look like this:
//
// -------------------------------------------
// | user func that is about to do a memop   |
// -------------------------------------------
// | __csan_load/store                       |
// -------------------------------------------
// | backtrace (assume __csan_load/store and |
// |            get_user_code_rip is inlined)|
// -------------------------------------------
//
// In the user program, __csan_load/store are inlined
// right before the corresponding read / write in the user code.
// the return addr of __csan_load/store is the rip for the read / write
CILKSAN_API
void __csan_load(csi_id_t load_id, const void *addr, int32_t size,
                 load_prop_t prop) {
  // TODO: Use alignment information.
  cilksan_assert(TOOL_INITIALIZED);
  if (!should_check()) {
    DBG_TRACE(DEBUG_MEMORY, "SKIP %s read %p\n", __FUNCTION__, addr);
    return;
  }
  if (!(*parallel_execution.head())) {
    DBG_TRACE(DEBUG_MEMORY, "SKIP %s read %p during serial execution\n",
              __FUNCTION__, addr);
    return;
  }

  CheckingRAII nocheck;
  // Record the address of this load.
  if (__builtin_expect(!load_pc[load_id], false))
    load_pc[load_id] = CALLERPC;

  DBG_TRACE(DEBUG_MEMORY, "%s read %p\n", __FUNCTION__, addr);
  // Record this read.
  CilkSanImpl.do_read(load_id, (uintptr_t)addr, size);
}

CILKSAN_API
void __csan_large_load(csi_id_t load_id, const void *addr, size_t size,
                       load_prop_t prop) {
  // TODO: Use alignment information.
  cilksan_assert(TOOL_INITIALIZED);
  if (!should_check()) {
    DBG_TRACE(DEBUG_MEMORY, "SKIP %s read %p\n", __FUNCTION__, addr);
    return;
  }
  if (!(*parallel_execution.head())) {
    DBG_TRACE(DEBUG_MEMORY, "SKIP %s read %p during serial execution\n",
              __FUNCTION__, addr);
    return;
  }

  CheckingRAII nocheck;
  // Record the address of this load.
  if (__builtin_expect(!load_pc[load_id], false))
    load_pc[load_id] = CALLERPC;

  DBG_TRACE(DEBUG_MEMORY, "%s read %p\n", __FUNCTION__, addr);
  // Record this read.
  CilkSanImpl.do_read(load_id, (uintptr_t)addr, size);
}

CILKSAN_API
void __csan_store(csi_id_t store_id, const void *addr, int32_t size,
                  store_prop_t prop) {
  // TODO: Use alignment information.
  cilksan_assert(TOOL_INITIALIZED);
  if (!should_check()) {
    DBG_TRACE(DEBUG_MEMORY, "SKIP %s wrote %p\n", __FUNCTION__, addr);
    return;
  }
  if (!(*parallel_execution.head())) {
    DBG_TRACE(DEBUG_MEMORY, "SKIP %s wrote %p during serial execution\n",
              __FUNCTION__, addr);
    return;
  }

  CheckingRAII nocheck;
  // Record the address of this store.
  if (__builtin_expect(!store_pc[store_id], false))
    store_pc[store_id] = CALLERPC;

  DBG_TRACE(DEBUG_MEMORY, "%s wrote %p\n", __FUNCTION__, addr);
  // Record this write.
  CilkSanImpl.do_write(store_id, (uintptr_t)addr, size);
}

CILKSAN_API
void __csan_large_store(csi_id_t store_id, const void *addr, size_t size,
                        store_prop_t prop) {
  // TODO: Use alignment information.
  cilksan_assert(TOOL_INITIALIZED);
  if (!should_check()) {
    DBG_TRACE(DEBUG_MEMORY, "SKIP %s wrote %p\n", __FUNCTION__, addr);
    return;
  }
  if (!(*parallel_execution.head())) {
    DBG_TRACE(DEBUG_MEMORY, "SKIP %s wrote %p during serial execution\n",
              __FUNCTION__, addr);
    return;
  }

  CheckingRAII nocheck;
  // Record the address of this store.
  if (__builtin_expect(!store_pc[store_id], false))
    store_pc[store_id] = CALLERPC;

  DBG_TRACE(DEBUG_MEMORY, "%s wrote %p\n", __FUNCTION__, addr);
  // Record this write.
  CilkSanImpl.do_write(store_id, (uintptr_t)addr, size);
}

CILKSAN_API
void __csi_after_alloca(const csi_id_t alloca_id, const void *addr,
                        size_t size, const alloca_prop_t prop) {
  cilksan_assert(TOOL_INITIALIZED);
  if (!should_check())
    return;

  if (stack_low_addr > (uintptr_t)addr)
    stack_low_addr = (uintptr_t)addr;

  CheckingRAII nocheck;
  // Record the PC for this alloca
  if (__builtin_expect(!alloca_pc[alloca_id], false))
    alloca_pc[alloca_id] = CALLERPC;

  DBG_TRACE(DEBUG_CALLBACK, "__csi_after_alloca(%ld)\n", alloca_id);

  // Record the alloca and clear the allocated portion of the shadow memory.
  CilkSanImpl.record_alloc((size_t) addr, size, 2 * alloca_id);
  CilkSanImpl.clear_shadow_memory((size_t)addr, size);
  CilkSanImpl.advance_stack_frame((uintptr_t)addr);
}

static std::unordered_map<uintptr_t, size_t> malloc_sizes;

CILKSAN_API
void __csan_after_allocfn(const csi_id_t allocfn_id, const void *addr,
                          size_t size, size_t num, size_t alignment,
                          const void *oldaddr, const allocfn_prop_t prop) {
  cilksan_assert(TOOL_INITIALIZED);
  if (!should_check())
    return;

  CheckingRAII nocheck;

  DBG_TRACE(DEBUG_CALLBACK,
            "__csan_after_allocfn(%ld, %s, addr = %p, size = %ld, oldaddr = %p)\n",
            allocfn_id, __csan_get_allocfn_str(prop), addr, size, oldaddr);

  // std::cerr << "Called memory function " << __csan_get_allocfn_str(prop) << "\n";

  // TODO: Use alignment information
  // Record the PC for this allocation-function call
  if (__builtin_expect(!allocfn_pc[allocfn_id], false))
    allocfn_pc[allocfn_id] = CALLERPC;
  if (__builtin_expect(allocfn_prop[allocfn_id].allocfn_ty == uint8_t(-1),
                       false))
    allocfn_prop[allocfn_id] = prop;

  size_t new_size = size * num;

  // If this allocation function operated on an old address -- e.g., a realloc
  // -- then update the memory at the old address as if it was freed.
  if (oldaddr) {
    auto iter = malloc_sizes.find((uintptr_t)oldaddr);
    if (oldaddr != addr) {
      if (new_size > 0) {
        // Record the new allocation.
        CilkSanImpl.record_alloc((size_t)addr, new_size, 2 * allocfn_id + 1);
        CilkSanImpl.clear_shadow_memory((size_t)addr, new_size);
        malloc_sizes.insert({(uintptr_t)addr, new_size});
      }

      if (iter != malloc_sizes.end()) {
        // Take note of the freeing of the old memory.
        CilkSanImpl.record_free((uintptr_t)oldaddr, iter->second, allocfn_id,
                                MAType_t::REALLOC);
        malloc_sizes.erase(iter);
      }
    } else {
      // We're simply adjusting the allocation at the same place.
      if (iter != malloc_sizes.end()) {
        size_t old_size = iter->second;
        if (old_size < new_size) {
          CilkSanImpl.clear_shadow_memory((size_t)addr + old_size,
                                          new_size - old_size);
        } else if (old_size > new_size) {
          // Take note of the effective free of the old space.
          CilkSanImpl.record_free((uintptr_t)oldaddr + new_size,
                                  old_size - new_size, allocfn_id,
                                  MAType_t::REALLOC);
        }
        CilkSanImpl.record_alloc((size_t)addr, new_size, 2 * allocfn_id + 1);
        malloc_sizes.erase(iter);
      }
      malloc_sizes.insert({(uintptr_t)addr, new_size});
    }
    return;
  }

  // For many memory allocation functions, including malloc and realloc, if the
  // requested size is 0, the behavior is implementation defined.  The function
  // might return nullptr, or return a non-null pointer that won't be used to
  // access memory.  We simply don't record an allocation of zero size.
  if (0 == size)
    return;

  // Record the new allocation.
  malloc_sizes.insert({(uintptr_t)addr, new_size});
  CilkSanImpl.record_alloc((size_t)addr, new_size, 2 * allocfn_id + 1);
  CilkSanImpl.clear_shadow_memory((size_t)addr, new_size);
}

// __csan_after_free is called after any call to free or delete.
CILKSAN_API
void __csan_after_free(const csi_id_t free_id, const void *ptr,
                       const free_prop_t prop) {
  cilksan_assert(TOOL_INITIALIZED);
  if (!should_check())
    return;

  CheckingRAII nocheck;

  // std::cerr << "Called memory function " << __csan_get_free_str(prop) << "\n";

  if (__builtin_expect(!free_pc[free_id], false))
    free_pc[free_id] = CALLERPC;

  auto iter = malloc_sizes.find((uintptr_t)ptr);
  if (iter != malloc_sizes.end()) {
    // cilksan_clear_shadow_memory((size_t)ptr, iter->second);

    // Treat a free as a write to all freed addresses.  This way the tool will
    // report a race if an operation tries to access a location that was freed
    // in parallel.
    CilkSanImpl.record_free((uintptr_t)ptr, iter->second, free_id,
                            MAType_t::FREE);
    malloc_sizes.erase(iter);
  }
}

#if CILKSAN_DYNAMIC

static std::map<uintptr_t, size_t> pages_to_clear;

// Flag to manage initialization of memory functions.  We need this flag because
// dlsym uses some of the memory functions we are trying to interpose, which
// means that calling dlysm directly will lead to infinite recursion and a
// segfault.  Fortunately, dlsym can make do with memory-allocation functions
// returning NULL, so we return NULL when we detect this inifinite recursion.
//
// This trick seems questionable, but it also seems to be standard practice.
// It's the same trick used by memusage.c in glibc, and there's little
// documentation on better tricks.
static int mem_initialized = 0;

// Pointer to real memory functions.
typedef void*(*malloc_t)(size_t);
static malloc_t real_malloc = NULL;

typedef void*(*calloc_t)(size_t, size_t);
static calloc_t real_calloc = NULL;

typedef void*(*realloc_t)(void*, size_t);
static realloc_t real_realloc = NULL;

typedef void(*free_t)(void*);
static free_t real_free = NULL;

typedef void*(*mmap_t)(void*, size_t, int, int, int, off_t);
static mmap_t real_mmap = NULL;

#if defined(_LARGEFILE64_SOURCE)
typedef void*(*mmap64_t)(void*, size_t, int, int, int, off64_t);
static mmap64_t real_mmap64 = NULL;
#endif // defined(_LARGEFILE64_SOURCE)

typedef int(*munmap_t)(void*, size_t);
static munmap_t real_munmap = NULL;

typedef void*(*mremap_t)(void*, size_t, size_t, int, ...);
static mremap_t real_mremap = NULL;

// Helper function to get real implementations of memory functions via dlsym.
static void initialize_memory_functions() {
  disable_checking();
  mem_initialized = -1;

  real_malloc = (malloc_t)dlsym(RTLD_NEXT, "malloc");
  char *error = dlerror();
  if (error != NULL)
    goto error_exit;

  real_calloc = (calloc_t)dlsym(RTLD_NEXT, "calloc");
  error = dlerror();
  if (error != NULL)
    goto error_exit;

  real_realloc = (realloc_t)dlsym(RTLD_NEXT, "realloc");
  error = dlerror();
  if (error != NULL)
    goto error_exit;

  real_free = (free_t)dlsym(RTLD_NEXT, "free");
  error = dlerror();
  if (error != NULL)
    goto error_exit;

  real_mmap = (mmap_t)dlsym(RTLD_NEXT, "mmap");
  error = dlerror();
  if (error != NULL)
    goto error_exit;

#if defined(_LARGEFILE64_SOURCE)
  real_mmap64 = (mmap64_t)dlsym(RTLD_NEXT, "mmap64");
  error = dlerror();
  if (error != NULL)
    goto error_exit;
#endif // defined(_LARGEFILE64_SOURCE)

  real_munmap = (munmap_t)dlsym(RTLD_NEXT, "munmap");
  error = dlerror();
  if (error != NULL)
    goto error_exit;

  real_mremap = (mremap_t)dlsym(RTLD_NEXT, "mremap");
  error = dlerror();
  if (error != NULL)
    goto error_exit;

  mem_initialized = 1;
  enable_checking();
  return;

 error_exit:
  fputs(error, err_io);
  fflush(err_io);
  abort();
  enable_checking();
  return;
}

// CILKSAN_API void* malloc(size_t s) {
//   // Don't try to init, since that needs malloc.
//   if (__builtin_expect(real_malloc == NULL, 0)) {
//     if (-1 == mem_initialized)
//       return NULL;
//     initialize_memory_functions();
//   }

//   disable_checking();
//   // align the allocation to simplify erasing from shadow mem
//   // uint64_t new_size = ALIGN_BY_NEXT_MAX_GRAIN_SIZE(s);
//   size_t new_size = ALIGN_FOR_MALLOC(s);
//   assert(s == new_size);
//   // call the real malloc
//   void *r = real_malloc(new_size);
//   enable_checking();

//   if (TOOL_INITIALIZED && should_check()) {
//     disable_checking();
//     malloc_sizes.insert({(uintptr_t)r, new_size});
//     // cilksan_clear_shadow_memory((size_t)r, (size_t)r+malloc_usable_size(r)-1);
//     cilksan_record_alloc((size_t)r, new_size, 0);
//     cilksan_clear_shadow_memory((size_t)r, new_size);
//     enable_checking();
//   }

//   return r;
// }

// CILKSAN_API void* calloc(size_t num, size_t s) {
//   if (__builtin_expect(real_calloc == NULL, 0)) {
//     if (-1 == mem_initialized)
//       return NULL;
//     initialize_memory_functions();
//   }

//   disable_checking();
//   void *r = real_calloc(num, s);
//   enable_checking();

//   if (TOOL_INITIALIZED && should_check()) {
//     disable_checking();
//     malloc_sizes.insert({(uintptr_t)r, s});
//     // cilksan_clear_shadow_memory((size_t)r, (size_t)r+malloc_usable_size(r)-1);
//     cilksan_record_alloc((size_t)r, num * s, 0);
//     cilksan_clear_shadow_memory((size_t)r, num * s);
//     enable_checking();
//   }

//   return r;
// }

// CILKSAN_API void free(void *ptr) {
//   if (__builtin_expect(real_free == NULL, 0)) {
//     if (-1 == mem_initialized)
//       return;
//     initialize_memory_functions();
//   }

//   disable_checking();
//   real_free(ptr);
//   enable_checking();

//   if (TOOL_INITIALIZED && should_check()) {
//     disable_checking();
//     auto iter = malloc_sizes.find((uintptr_t)ptr);
//     if (iter != malloc_sizes.end()) {
//       // cilksan_clear_shadow_memory((size_t)ptr, iter->second);

//       // Treat a free as a write to all freed addresses.  This way, the tool
//       // will report a race if an operation tries to access a location that was
//       // freed in parallel.
//       cilksan_do_write(UNKNOWN_CSI_ID, (uintptr_t)ptr, iter->second);
//       malloc_sizes.erase(iter);
//     }
//     enable_checking();
//   }
// }

// CILKSAN_API void* realloc(void *ptr, size_t s) {
//   if (__builtin_expect(real_realloc == NULL, 0)) {
//     if (-1 == mem_initialized)
//       return NULL;
//     initialize_memory_functions();
//   }

//   disable_checking();
//   void *r = real_realloc(ptr, s);
//   enable_checking();

//   if (TOOL_INITIALIZED && should_check()) {
//     disable_checking();
//     // Treat the old pointer ptr as freed and the new pointer r as freshly
//     // malloc'd.
//     auto iter = malloc_sizes.find((uintptr_t)ptr);
//     if (iter != malloc_sizes.end()) {
//       cilksan_do_write(UNKNOWN_CSI_ID, (uintptr_t)ptr, iter->second);
//       malloc_sizes.erase(iter);
//     }
//     malloc_sizes.insert({(uintptr_t)r, s});
//     // cilksan_clear_shadow_memory((size_t)r, (size_t)r+malloc_usable_size(r)-1);
//     cilksan_record_alloc((size_t)r, s, 0);
//     cilksan_clear_shadow_memory((size_t)r, s);
//     enable_checking();
//   }

//   return r;
// }

CILKSAN_API
void *mmap(void *start, size_t len, int prot, int flags, int fd, off_t offset) {
  if (__builtin_expect(real_mmap == NULL, 0)) {
    if (-1 == mem_initialized)
      return NULL;
    initialize_memory_functions();
  }

  disable_checking();
  void *r = real_mmap(start, len, prot, flags, fd, offset);
  enable_checking();

  if (TOOL_INITIALIZED && should_check()) {
    CheckingRAII nocheck;
    CilkSanImpl.record_alloc((size_t)r, len, 0);
    CilkSanImpl.clear_shadow_memory((size_t)r, len);
    pages_to_clear.insert({(uintptr_t)r, len});
    if (!(flags & MAP_ANONYMOUS))
      // This mmap is backed by a file.  Initialize the shadow memory with a
      // write to the page.
      CilkSanImpl.do_write(UNKNOWN_CSI_ID, (uintptr_t)r, len);
  }

  return r;
}

#if defined(_LARGEFILE64_SOURCE)
CILKSAN_API
void *mmap64(void *start, size_t len, int prot, int flags, int fd, off64_t offset) {
  if (__builtin_expect(real_mmap64 == NULL, 0)) {
    if (-1 == mem_initialized)
      return NULL;
    initialize_memory_functions();
  }

  disable_checking();
  void *r = real_mmap64(start, len, prot, flags, fd, offset);
  enable_checking();

  if (TOOL_INITIALIZED && should_check()) {
    CheckingRAII nocheck;
    CilkSanImpl.record_alloc((size_t)r, len, 0);
    CilkSanImpl.clear_shadow_memory((size_t)r, len);
    pages_to_clear.insert({(uintptr_t)r, len});
    if (!(flags & MAP_ANONYMOUS))
      // This mmap is backed by a file.  Initialize the shadow memory with a
      // write to the page.
      CilkSanImpl.do_write(UNKNOWN_CSI_ID, (uintptr_t)r, len);
  }

  return r;
}
#endif // defined(_LARGEFILE64_SOURCE)

CILKSAN_API
int munmap(void *start, size_t len) {
  if (__builtin_expect(real_munmap == NULL, 0)) {
    if (-1 == mem_initialized)
      return -1;
    initialize_memory_functions();
  }

  disable_checking();
  int result = real_munmap(start, len);
  enable_checking();

  if (TOOL_INITIALIZED && should_check() && (0 == result)) {
    CheckingRAII nocheck;
    auto first_page = pages_to_clear.lower_bound((uintptr_t)start);
    auto last_page = pages_to_clear.upper_bound((uintptr_t)start + len);
    for (auto curr_page = first_page; curr_page != last_page; ++curr_page) {
      // TODO: Treat unmap more like free and record a write operation on the
      // page.  Need to take care only to write pages that have content in the
      // shadow memory.  Otherwise, if the application mmap's more virtual
      // memory than physical memory, then the writes that model page unmapping
      // can blow out physical memory.
      CilkSanImpl.clear_shadow_memory((size_t)curr_page->first, curr_page->second);
      // CilkSanImpl.do_write(UNKNOWN_CSI_ID, curr_page->first, curr_page->second);
    }
    pages_to_clear.erase(first_page, last_page);
  }

  return result;
}

CILKSAN_API
void *mremap(void *start, size_t old_len, size_t len, int flags, ...) {
#if defined(MREMAP_FIXED)
  va_list ap;
  va_start (ap, flags);
  void *newaddr = (flags & MREMAP_FIXED) ? va_arg (ap, void *) : NULL;
  va_end (ap);
#endif // defined(MREMAP_FIXED)

  if (__builtin_expect(real_mremap == NULL, 0)) {
    if (-1 == mem_initialized)
      return NULL;
    initialize_memory_functions();
  }

  disable_checking();
#if defined(MREMAP_FIXED)
  void *r = real_mremap(start, old_len, len, flags, newaddr);
#else
  void *r = real_mremap(start, old_len, len, flags);
#endif // defined(MREMAP_FIXED)
  enable_checking();

  if (TOOL_INITIALIZED && should_check()) {
    CheckingRAII nocheck;
    auto iter = pages_to_clear.find((uintptr_t)start);
    if (iter != pages_to_clear.end()) {
      // TODO: Treat unmap more like free and record a write operation on the
      // page.  Need to take care only to write pages that have content in the
      // shadow memory.  Otherwise, if the application mmap's more virtual
      // memory than physical memory, then the writes that model page unmapping
      // can blow out physical memory.
      CilkSanImpl.clear_shadow_memory((size_t)iter->first, iter->second);
      // cilksan_do_write(UNKNOWN_CSI_ID, iter->first, iter->second);
      pages_to_clear.erase(iter);
    }
    // Record the new mapping.
    CilkSanImpl.record_alloc((size_t)r, len, 0);
    CilkSanImpl.clear_shadow_memory((size_t)r, len);
    pages_to_clear.insert({(uintptr_t)r, len});
  }

  return r;
}

#endif  // CILKSAN_DYNAMIC
