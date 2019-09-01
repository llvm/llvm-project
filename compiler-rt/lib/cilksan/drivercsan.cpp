// #include <cilk/hyperobject_base.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <execinfo.h>
// #include <internal/abi.h>
#include <malloc.h>
#include <unistd.h>
#include <unordered_map>
#include <sys/mman.h>

#if CILKSAN_DYNAMIC
#include <dlfcn.h>
#endif  // CILKSAN_DYNAMIC

#include "cilksan_internal.h"
#include "debug_util.h"
#include "mem_access.h"
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
extern uintptr_t *load_pc;
extern uintptr_t *store_pc;
extern uintptr_t *alloca_pc;
extern uintptr_t *allocfn_pc;
static csi_id_t total_call = 0;
static csi_id_t total_spawn = 0;
static csi_id_t total_load = 0;
static csi_id_t total_store = 0;
static csi_id_t total_alloca = 0;
static csi_id_t total_allocfn = 0;

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

  char *e = getenv("CILK_NWORKERS");
  if (!e || 0 != strcmp(e, "1")) {
    // fprintf(err_io, "Setting CILK_NWORKERS to be 1\n");
    if( setenv("CILK_NWORKERS", "1", 1) ) {
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
    if( setenv("CILK_FORCE_REDUCE", "1", 1) ) {
      fprintf(err_io, "Error setting CILK_FORCE_REDUCE to be 1\n");
      exit(1);
    }
  }
}

CILKSAN_API void __csi_init() {
  // This method should only be called once.
  assert(!TOOL_INITIALIZED && "__csi_init() called multiple times.");

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
  if (counts.num_load)
    grow_pc_table(load_pc, total_load, counts.num_load);
  if (counts.num_store)
    grow_pc_table(store_pc, total_store, counts.num_store);
  if (counts.num_alloca)
    grow_pc_table(alloca_pc, total_alloca, counts.num_alloca);
  if (counts.num_allocfn)
    grow_pc_table(allocfn_pc, total_allocfn, counts.num_allocfn);
}

// invoked whenever a function enters; no need for this
CILKSAN_API void __csan_func_entry(const csi_id_t func_id,
                                   void *bp, void *sp,
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
  const csan_source_loc_t *srcloc = __csan_get_func_source_loc(func_id);
  DBG_TRACE(DEBUG_CALLBACK, "__csan_func_entry(%d) at %s (%s:%d)\n",
            func_id,
            srcloc->name, srcloc->filename,
            srcloc->line_number);
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
  CilkSanImpl.do_enter_begin();
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
  const csan_source_loc_t *srcloc = __csan_get_func_exit_source_loc(func_exit_id);
  DBG_TRACE(DEBUG_CALLBACK, "__csan_func_exit(%ld, %ld) at %s (%s:%d)\n",
            func_exit_id, func_id,
            srcloc->name, srcloc->filename,
            srcloc->line_number);

  // if (prop.may_spawn) {
    // Update the tool for leaving a Cilk function.
    CilkSanImpl.do_leave_begin();
    CilkSanImpl.do_leave_end();
  // }

  // Pop both local copies of the parallel-execution state.
  parallel_execution.pop();
  parallel_execution.pop();

  CilkSanImpl.pop_stack_frame();

  // XXX Let's focus on Cilk function for now; maybe put it back later
  // cilksan_do_function_exit();
}

CILKSAN_API void __csan_before_call(const csi_id_t call_id,
                                    const csi_id_t func_id,
                                    unsigned suppression_count,
                                    const call_prop_t prop) {
  if (!should_check())
    return;

  CheckingRAII nocheck;
  DBG_TRACE(DEBUG_CALLBACK, "__csi_before_call(%ld, %ld)\n",
            call_id, func_id);

  // Record the address of this call site.
  if (!call_pc[call_id])
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
  DBG_TRACE(DEBUG_CALLBACK, "__csi_after_call(%ld, %ld)\n",
            call_id, func_id);

  // Pop any suppressions.
  for (unsigned i = 0; i < suppression_count; ++i)
    suppressions.pop();
  suppression_counts.pop();

  // Pop the call off of the call stack.
  CilkSanImpl.record_call_return(call_id, CALL);
}

CILKSAN_API void __csan_detach(const csi_id_t detach_id) {
  if (!should_check())
    return;

  CheckingRAII nocheck;
  DBG_TRACE(DEBUG_CALLBACK, "__csan_detach(%ld)\n",
            detach_id);
  cilksan_assert(last_event == NONE);
  WHEN_CILKSAN_DEBUG(last_event = SPAWN_PREPARE);
  WHEN_CILKSAN_DEBUG(last_event = NONE);

  // Record the address of this detach.
  if (!spawn_pc[detach_id])
    spawn_pc[detach_id] = CALLERPC;

  // Update the parallel-execution state to reflect this detach.  Essentially,
  // this notes the change of peer sets.
  *parallel_execution.head() = 1;

  // Push the detach onto the call stack.
  CilkSanImpl.record_call(detach_id, SPAWN);
}

CILKSAN_API void __csan_task(const csi_id_t task_id, const csi_id_t detach_id,
                             void *bp, void *sp) {
  if (!should_check())
    return;
  // fprintf(stderr, "__csan_task: bp = %p, sp = %p\n",
  //         (uintptr_t)bp, (uintptr_t)sp);

  if (stack_low_addr > (uintptr_t)sp)
    stack_low_addr = (uintptr_t)sp;

  CheckingRAII nocheck;
  DBG_TRACE(DEBUG_CALLBACK, "__csan_task(%ld, %ld)\n",
            task_id, detach_id);
  WHEN_CILKSAN_DEBUG(last_event = NONE);

  // Propagate the parallel-execution state to the child.
  uint8_t current_pe = *parallel_execution.head();
  // First we push the pe value on function entry.
  parallel_execution.push();
  *parallel_execution.head() = current_pe;
  // We push a second copy to update aggressively on detaches.
  parallel_execution.push();
  *parallel_execution.head() = current_pe;

  CilkSanImpl.push_stack_frame((uintptr_t)bp, (uintptr_t)sp);

  // Update tool for entering detach-helper function and performing detach.
  CilkSanImpl.do_enter_helper_begin();
  CilkSanImpl.do_enter_end((uintptr_t)sp);
  CilkSanImpl.do_detach_begin();
  CilkSanImpl.do_detach_end();
}

CILKSAN_API void __csan_task_exit(const csi_id_t task_exit_id,
                                  const csi_id_t task_id,
                                  const csi_id_t detach_id) {
  if (!should_check())
    return;
  CheckingRAII nocheck;
  DBG_TRACE(DEBUG_CALLBACK, "__csan_task_exit(%ld, %ld, %ld)\n",
            task_exit_id, task_id, detach_id);

  // Update tool for leaving a detach-helper function.
  CilkSanImpl.do_leave_begin();
  CilkSanImpl.do_leave_end();

  // Pop the parallel-execution state.
  parallel_execution.pop();
  parallel_execution.pop();

  CilkSanImpl.pop_stack_frame();
}

CILKSAN_API void __csan_detach_continue(const csi_id_t detach_continue_id,
                                        const csi_id_t detach_id) {
  if (!should_check())
    return;
  CheckingRAII nocheck;
  DBG_TRACE(DEBUG_CALLBACK, "__csan_detach_continue(%ld)\n",
            detach_id);

  CilkSanImpl.record_call_return(detach_id, SPAWN);

  if (last_event == LEAVE_FRAME_OR_HELPER)
    CilkSanImpl.do_leave_end();

  WHEN_CILKSAN_DEBUG(last_event = NONE);
}

CILKSAN_API void __csan_sync(csi_id_t sync_id) {
  if (!should_check())
    return;
  CheckingRAII nocheck;
  cilksan_assert(TOOL_INITIALIZED);

  // Because this is a serial tool, we can safely perform all operations related
  // to a sync.
  CilkSanImpl.do_sync_begin();
  CilkSanImpl.do_sync_end();

  // Restore the parallel-execution state to that of the function/task entry.
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
void __csan_load(csi_id_t load_id, void *addr, int32_t size, load_prop_t prop) {
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
  if (!load_pc[load_id])
    load_pc[load_id] = CALLERPC;

  DBG_TRACE(DEBUG_MEMORY, "%s read %p\n", __FUNCTION__, addr);
  // Record this read.
  CilkSanImpl.do_read(load_id, (uintptr_t)addr, size);
}

CILKSAN_API
void __csan_large_load(csi_id_t load_id, void *addr, size_t size,
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
  if (!load_pc[load_id])
    load_pc[load_id] = CALLERPC;

  DBG_TRACE(DEBUG_MEMORY, "%s read %p\n", __FUNCTION__, addr);
  // Record this read.
  CilkSanImpl.do_read(load_id, (uintptr_t)addr, size);
}

CILKSAN_API
void __csan_store(csi_id_t store_id, void *addr, int32_t size, store_prop_t prop) {
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
  if (!store_pc[store_id])
    store_pc[store_id] = CALLERPC;

  DBG_TRACE(DEBUG_MEMORY, "%s wrote %p\n", __FUNCTION__, addr);
  // Record this write.
  CilkSanImpl.do_write(store_id, (uintptr_t)addr, size);
}

CILKSAN_API
void __csan_large_store(csi_id_t store_id, void *addr, size_t size,
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
  if (!store_pc[store_id])
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
  if (!alloca_pc[alloca_id])
    alloca_pc[alloca_id] = CALLERPC;

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

  // std::cerr << "Called memory function " << __csan_get_allocfn_str(prop) << "\n";

  // TODO: Use alignment information
  // Record the PC for this allocation-function call
  if (!allocfn_pc[allocfn_id])
    allocfn_pc[allocfn_id] = CALLERPC;

  // If this allocation function operated on an old address -- e.g., a realloc
  // -- then update the memory at the old address as if it was freed.
  if (oldaddr) {
    auto iter = malloc_sizes.find((uintptr_t)oldaddr);
    if (iter != malloc_sizes.end()) {
      CilkSanImpl.do_write(UNKNOWN_CSI_ID, (uintptr_t)oldaddr, iter->second);
      malloc_sizes.erase(iter);
    }
  }
  // Record the new allocation.
  malloc_sizes.insert({(uintptr_t)addr, size * num});
  CilkSanImpl.record_alloc((size_t)addr, size * num, 2 * allocfn_id + 1);
  CilkSanImpl.clear_shadow_memory((size_t)addr, size * num);
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

  auto iter = malloc_sizes.find((uintptr_t)ptr);
  if (iter != malloc_sizes.end()) {
    // cilksan_clear_shadow_memory((size_t)ptr, iter->second);

    // Treat a free as a write to all freed addresses.  This way the tool will
    // report a race if an operation tries to access a location that was freed
    // in parallel.
    CilkSanImpl.do_write(UNKNOWN_CSI_ID, (uintptr_t)ptr, iter->second);
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

typedef void*(*mmap64_t)(void*, size_t, int, int, int, off64_t);
static mmap64_t real_mmap64 = NULL;

typedef int(*munmap_t)(void*, size_t);
static munmap_t real_munmap = NULL;

typedef void*(*mremap_t)(void*, size_t, size_t, int, void*);
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

  real_mmap64 = (mmap64_t)dlsym(RTLD_NEXT, "mmap64");
  error = dlerror();
  if (error != NULL)
    goto error_exit;

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
  va_list ap;
  va_start (ap, flags);
  void *newaddr = (flags & MREMAP_FIXED) ? va_arg (ap, void *) : NULL;
  va_end (ap);

  if (__builtin_expect(real_mremap == NULL, 0)) {
    if (-1 == mem_initialized)
      return NULL;
    initialize_memory_functions();
  }

  disable_checking();
  void *r = real_mremap(start, old_len, len, flags, newaddr);
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
