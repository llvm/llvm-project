// #include <cilk/hyperobject_base.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <dlfcn.h>
#include <execinfo.h>
// #include <internal/abi.h>
#include <malloc.h>
#include <unistd.h>
#include <unordered_map>
#include <sys/mman.h>

#include "cilksan_internal.h"
#include "debug_util.h"
#include "mem_access.h"
#include "stack.h"

#define CALLERPC ((uintptr_t)__builtin_return_address(0))

#define CILKSAN_API extern "C" __attribute__((visibility("default")))

// global var: FILE io used to print error messages
FILE *err_io;

// Defined in print_addr.cpp
extern void read_proc_maps();
extern void delete_proc_maps();
extern void print_addr(FILE *f, void *a);
// declared in cilksan; for debugging only
#if CILKSAN_DEBUG
extern enum EventType_t last_event;
#endif

// Defined in cilksan.cpp
extern call_stack_t call_stack;
extern Stack_t<uintptr_t> sp_stack;
// Defined in print_addr.cpp
extern uintptr_t *call_pc;
extern uintptr_t *spawn_pc;
extern uintptr_t *load_pc;
extern uintptr_t *store_pc;
extern uintptr_t *alloca_pc;
static csi_id_t total_call = 0;
static csi_id_t total_spawn = 0;
static csi_id_t total_load = 0;
static csi_id_t total_store = 0;
static csi_id_t total_alloca = 0;

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
  return (instrumentation && checking_disabled == 0);
}

// called upon process exit
static void csan_destroy(void) {
  // fprintf(err_io, "csan_destroy called.\n");
  disable_instrumentation();
  disable_checking();
  cilksan_deinit();
  // std::cerr << "call_stack.size " << call_stack.size() << std::endl;
  fflush(stdout);
  delete_proc_maps();
}

static void init_internal() {
  read_proc_maps();
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
  if (TOOL_INITIALIZED) assert(0);

  atexit(csan_destroy);
  init_internal();
  // moved this later when we enter the first Cilk frame
  // cilksan_init();
  // enable_instrumentation();
  TOOL_INITIALIZED = true;
  // fprintf(err_io, "tsan_init called.\n");
}

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
                      const csan_instrumentation_counts_t counts)
{
  disable_checking();
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
  enable_checking();
}

// invoked whenever a function enters; no need for this
CILKSAN_API void __csan_func_entry(const csi_id_t func_id,
                                   void *sp,
                                   const func_prop_t prop) {
  const csan_source_loc_t *srcloc = __csan_get_func_source_loc(func_id);
  DBG_TRACE(DEBUG_CALLBACK, "__csan_func_entry(%d) at %s (%s:%d)\n",
            func_id,
            srcloc->name, srcloc->filename,
            srcloc->line_number);
  cilksan_assert(TOOL_INITIALIZED);
  if (!prop.may_spawn)
    // Ignore entry calls into non-Cilk functions.
    return;

  disable_checking();
  static bool first_call = true;
  if (first_call) {
    cilksan_init();
    first_call = false;
  }
  // Record high location of the stack for this frame.
  sp_stack.push();
  *sp_stack.head() = (uintptr_t)sp;
  // Record low location of the stack for this frame.  This value will be
  // updated by reads and writes to the stack.
  sp_stack.push();
  *sp_stack.head() = (uintptr_t)sp;

  // Update the tool for entering a Cilk function.
  cilksan_do_enter_begin();
  cilksan_do_enter_end((uintptr_t)sp);
  enable_instrumentation();
  enable_checking();
}

CILKSAN_API void __csan_func_exit(const csi_id_t func_exit_id,
                                  const csi_id_t func_id,
                                  const func_exit_prop_t prop) {
  cilksan_assert(TOOL_INITIALIZED);
  if (prop.may_spawn) {
    disable_checking();
    // Update the tool for leaving a Cilk function.
    cilksan_do_leave_begin();
    cilksan_do_leave_end();

    // Pop stack pointers.
    uintptr_t low_stack = *sp_stack.head();
    sp_stack.pop();
    uintptr_t high_stack = *sp_stack.head();
    sp_stack.pop();
    assert(low_stack <= high_stack);
    // Clear shadow memory of stack locations.
    if (low_stack != high_stack)
      cilksan_clear_shadow_memory(low_stack, high_stack - low_stack);
    enable_checking();
  }
  // XXX Let's focus on Cilk function for now; maybe put it back later
  // cilksan_do_function_exit();
}

CILKSAN_API void __csi_before_call(const csi_id_t call_id,
                                   const csi_id_t func_id,
                                   const call_prop_t prop) {
  DBG_TRACE(DEBUG_CALLBACK, "__csi_before_call(%ld, %ld)\n",
            call_id, func_id);

  disable_checking();
  // Record the address of this call site.
  if (!call_pc[call_id])
    call_pc[call_id] = CALLERPC;
  // Push the call onto the call stack.
  call_stack.push(CallID_t(CALL, call_id));
  enable_checking();
}

CILKSAN_API void __csi_after_call(const csi_id_t call_id,
                                  const csi_id_t func_id,
                                  const call_prop_t prop) {
  DBG_TRACE(DEBUG_CALLBACK, "__csi_after_call(%ld, %ld)\n",
            call_id, func_id);
  // Pop the call off of the call stack.
  disable_checking();
  assert(call_stack.tail->id == CallID_t(CALL, call_id) &&
         "ERROR: after_call encountered without corresponding before_call");
  call_stack.pop();
  enable_checking();
}

CILKSAN_API void __csan_detach(const csi_id_t detach_id) {
  DBG_TRACE(DEBUG_CALLBACK, "__csan_detach(%ld)\n",
            detach_id);
  disable_checking();
  cilksan_assert(last_event == NONE);
  WHEN_CILKSAN_DEBUG(last_event = SPAWN_PREPARE);
  WHEN_CILKSAN_DEBUG(last_event = NONE);
  // Record the address of this detach.
  if (!spawn_pc[detach_id])
    spawn_pc[detach_id] = CALLERPC;
  // Push the detach onto the call stack.
  call_stack.push(CallID_t(SPAWN, detach_id));
  enable_checking();
}

CILKSAN_API void __csan_task(const csi_id_t task_id, const csi_id_t detach_id,
                            void *sp) {
  WHEN_CILKSAN_DEBUG(last_event = NONE);
  disable_checking();
  // Record high location of the stack for this frame.
  sp_stack.push();
  *sp_stack.head() = (uintptr_t)sp;
  // Record low location of the stack for this frame.  This value will be
  // updated by reads and writes to the stack.
  sp_stack.push();
  *sp_stack.head() = (uintptr_t)sp;

  // Update tool for entering detach-helper function and performing detach.
  cilksan_do_enter_helper_begin();
  cilksan_do_enter_end((uintptr_t)sp);
  cilksan_do_detach_begin();
  cilksan_do_detach_end();
  enable_checking();
}

CILKSAN_API void __csan_task_exit(const csi_id_t task_exit_id,
                                  const csi_id_t task_id,
                                  const csi_id_t detach_id) {
  disable_checking();
  // Update tool for leaving a detach-helper function.
  cilksan_do_leave_begin();
  cilksan_do_leave_end();

  // Pop stack pointers.
  uintptr_t low_stack = *sp_stack.head();
  sp_stack.pop();
  uintptr_t high_stack = *sp_stack.head();
  sp_stack.pop();
  assert(low_stack <= high_stack);
  // Clear shadow memory of stack locations.
  if (low_stack != high_stack)
    cilksan_clear_shadow_memory(low_stack, high_stack - low_stack);
  enable_checking();
}

CILKSAN_API void __csan_detach_continue(const csi_id_t detach_continue_id,
                                        const csi_id_t detach_id) {
  DBG_TRACE(DEBUG_CALLBACK, "__csan_detach_continue(%ld)\n",
            detach_id);
  disable_checking();
  assert(call_stack.tail->id == CallID_t(SPAWN, detach_id) &&
         "ERROR: detach_continue encountered without corresponding detach");
  call_stack.pop();

  if (last_event == LEAVE_FRAME_OR_HELPER)
    cilksan_do_leave_end();
  WHEN_CILKSAN_DEBUG(last_event = NONE);
  enable_checking();
}

CILKSAN_API void __csan_sync(csi_id_t sync_id) {
  disable_checking();
  cilksan_assert(TOOL_INITIALIZED);
  // Because this is a serial tool, we can safely perform all operations related
  // to a sync.
  cilksan_do_sync_begin();
  cilksan_do_sync_end();
  enable_checking();
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
  if (should_check()) {
    disable_checking();
    // Record the address of this load.
    if (!load_pc[load_id])
      load_pc[load_id] = CALLERPC;

    DBG_TRACE(DEBUG_MEMORY, "%s read %p\n", __FUNCTION__, addr);
    // Record this read.
    cilksan_do_read(load_id, (uintptr_t)addr, size);
    enable_checking();
  } else {
    DBG_TRACE(DEBUG_MEMORY, "SKIP %s read %p\n", __FUNCTION__, addr);
  }
}

CILKSAN_API
void __csan_large_load(csi_id_t load_id, void *addr, size_t size,
                       load_prop_t prop) {
  // TODO: Use alignment information.
  cilksan_assert(TOOL_INITIALIZED);
  if (should_check()) {
    disable_checking();
    // Record the address of this load.
    if (!load_pc[load_id])
      load_pc[load_id] = CALLERPC;

    DBG_TRACE(DEBUG_MEMORY, "%s read %p\n", __FUNCTION__, addr);
    // Record this read.
    cilksan_do_read(load_id, (uintptr_t)addr, size);
    enable_checking();
  } else {
    DBG_TRACE(DEBUG_MEMORY, "SKIP %s read %p\n", __FUNCTION__, addr);
  }
}

CILKSAN_API
void __csan_store(csi_id_t store_id, void *addr, int32_t size, store_prop_t prop) {
  // TODO: Use alignment information.
  cilksan_assert(TOOL_INITIALIZED);
  if (should_check()) {
    disable_checking();
    // Record the address of this store.
    if (!store_pc[store_id])
      store_pc[store_id] = CALLERPC;

    DBG_TRACE(DEBUG_MEMORY, "%s wrote %p\n", __FUNCTION__, addr);
    // Record this write.
    cilksan_do_write(store_id, (uintptr_t)addr, size);
    enable_checking();
  } else {
    DBG_TRACE(DEBUG_MEMORY, "SKIP %s wrote %p\n", __FUNCTION__, addr);
  }
}

CILKSAN_API
void __csan_large_store(csi_id_t store_id, void *addr, size_t size,
                        store_prop_t prop) {
  // TODO: Use alignment information.
  cilksan_assert(TOOL_INITIALIZED);
  if (should_check()) {
    disable_checking();
    // Record the address of this store.
    if (!store_pc[store_id])
      store_pc[store_id] = CALLERPC;

    DBG_TRACE(DEBUG_MEMORY, "%s wrote %p\n", __FUNCTION__, addr);
    // Record this write.
    cilksan_do_write(store_id, (uintptr_t)addr, size);
    enable_checking();
  } else {
    DBG_TRACE(DEBUG_MEMORY, "SKIP %s wrote %p\n", __FUNCTION__, addr);
  }
}

CILKSAN_API
void __csi_after_alloca(const csi_id_t alloca_id, const void *addr, uint64_t total_size, uint64_t isStaticAlloca) {
  if (TOOL_INITIALIZED && should_check()) {
    disable_checking();
    if (!alloca_pc[alloca_id])
      alloca_pc[alloca_id] = CALLERPC;
    cilksan_record_alloc((size_t) addr, total_size, alloca_id + 1);
    cilksan_clear_shadow_memory((size_t)addr, total_size);
    enable_checking();
  }
}

static std::unordered_map<uintptr_t, size_t> malloc_sizes;
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

CILKSAN_API void* malloc(size_t s) {
  // Don't try to init, since that needs malloc.
  if (__builtin_expect(real_malloc == NULL, 0)) {
    if (-1 == mem_initialized)
      return NULL;
    initialize_memory_functions();
  }

  disable_checking();
  // align the allocation to simplify erasing from shadow mem
  // uint64_t new_size = ALIGN_BY_NEXT_MAX_GRAIN_SIZE(s);
  size_t new_size = ALIGN_FOR_MALLOC(s);
  assert(s == new_size);
  // call the real malloc
  void *r = real_malloc(new_size);
  enable_checking();

  if (TOOL_INITIALIZED && should_check()) {
    disable_checking();
    malloc_sizes.insert({(uintptr_t)r, new_size});
    // cilksan_clear_shadow_memory((size_t)r, (size_t)r+malloc_usable_size(r)-1);
    cilksan_record_alloc((size_t)r, new_size, 0);
    cilksan_clear_shadow_memory((size_t)r, new_size);
    enable_checking();
  }

  return r;
}

CILKSAN_API void* calloc(size_t num, size_t s) {
  if (__builtin_expect(real_calloc == NULL, 0)) {
    if (-1 == mem_initialized)
      return NULL;
    initialize_memory_functions();
  }

  disable_checking();
  void *r = real_calloc(num, s);
  enable_checking();

  if (TOOL_INITIALIZED && should_check()) {
    disable_checking();
    malloc_sizes.insert({(uintptr_t)r, s});
    // cilksan_clear_shadow_memory((size_t)r, (size_t)r+malloc_usable_size(r)-1);
    cilksan_clear_shadow_memory((size_t)r, s);
    enable_checking();
  }

  return r;
}

CILKSAN_API void free(void *ptr) {
  if (__builtin_expect(real_free == NULL, 0)) {
    if (-1 == mem_initialized)
      return;
    initialize_memory_functions();
  }

  disable_checking();
  real_free(ptr);
  enable_checking();

  if (TOOL_INITIALIZED && should_check()) {
    disable_checking();
    auto iter = malloc_sizes.find((uintptr_t)ptr);
    if (iter != malloc_sizes.end()) {
      // cilksan_clear_shadow_memory((size_t)ptr, iter->second);

      // Treat a free as a write to all freed addresses.  This way, the tool
      // will report a race if an operation tries to access a location that was
      // freed in parallel.
      cilksan_do_write(UNKNOWN_CSI_ID, (uintptr_t)ptr, iter->second);
      malloc_sizes.erase(iter);
    }
    enable_checking();
  }
}

CILKSAN_API void* realloc(void *ptr, size_t s) {
  if (__builtin_expect(real_realloc == NULL, 0)) {
    if (-1 == mem_initialized)
      return NULL;
    initialize_memory_functions();
  }

  disable_checking();
  void *r = real_realloc(ptr, s);
  enable_checking();

  if (TOOL_INITIALIZED && should_check()) {
    disable_checking();
    // Treat the old pointer ptr as freed and the new pointer r as freshly
    // malloc'd.
    auto iter = malloc_sizes.find((uintptr_t)ptr);
    if (iter != malloc_sizes.end()) {
      cilksan_do_write(UNKNOWN_CSI_ID, (uintptr_t)ptr, iter->second);
      malloc_sizes.erase(iter);
    }
    malloc_sizes.insert({(uintptr_t)r, s});
    // cilksan_clear_shadow_memory((size_t)r, (size_t)r+malloc_usable_size(r)-1);
    cilksan_clear_shadow_memory((size_t)r, s);
    enable_checking();
  }

  return r;
}

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
    disable_checking();
    pages_to_clear.insert({(uintptr_t)r, len});
    if (!(flags & MAP_ANONYMOUS))
      // This mmap is backed by a file.  Initialize the shadow memory with a
      // write to the page.
      cilksan_do_write(UNKNOWN_CSI_ID, (uintptr_t)r, len);
    enable_checking();
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
    disable_checking();
    pages_to_clear.insert({(uintptr_t)r, len});
    if (!(flags & MAP_ANONYMOUS))
      // This mmap is backed by a file.  Initialize the shadow memory with a
      // write to the page.
      cilksan_do_write(UNKNOWN_CSI_ID, (uintptr_t)r, len);
    enable_checking();
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
    disable_checking();
    auto first_page = pages_to_clear.lower_bound((uintptr_t)start);
    auto last_page = pages_to_clear.upper_bound((uintptr_t)start + len);
    for (auto curr_page = first_page; curr_page != last_page; ++curr_page) {
      // TODO: Treat unmap more like free and record a write operation on the
      // page.  Need to take care only to write pages that have content in the
      // shadow memory.  Otherwise, if the application mmap's more virtual
      // memory than physical memory, then the writes that model page unmapping
      // can blow out physical memory.
      cilksan_clear_shadow_memory((size_t)curr_page->first, curr_page->second);
      // cilksan_do_write(UNKNOWN_CSI_ID, curr_page->first, curr_page->second);
    }
    pages_to_clear.erase(first_page, last_page);
    enable_checking();
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
    disable_checking();
    auto iter = pages_to_clear.find((uintptr_t)start);
    if (iter != pages_to_clear.end()) {
      // TODO: Treat unmap more like free and record a write operation on the
      // page.  Need to take care only to write pages that have content in the
      // shadow memory.  Otherwise, if the application mmap's more virtual
      // memory than physical memory, then the writes that model page unmapping
      // can blow out physical memory.
      cilksan_clear_shadow_memory((size_t)iter->first, iter->second);
      // cilksan_do_write(UNKNOWN_CSI_ID, iter->first, iter->second);
      pages_to_clear.erase(iter);
    }
    // Record the new mapping.
    pages_to_clear.insert({(uintptr_t)r, len});
    enable_checking();
  }

  return r;
}

// typedef void(*__cilkrts_hyper_create_t)(__cilkrts_hyperobject_base *);
// static __cilkrts_hyper_create_t real___cilkrts_hyper_create = NULL;
// CILKSAN_API
// void __cilkrts_hyper_create(__cilkrts_hyperobject_base *hb) {
//   disable_checking();
//   if (real___cilkrts_hyper_create == NULL) {
//     real___cilkrts_hyper_create =
//       (__cilkrts_hyper_create_t)dlsym(RTLD_NEXT, "__cilkrts_hyper_create");
//     char *error = dlerror();
//     if (error != NULL) {
//       fputs(error, err_io);
//       fflush(err_io);
//       abort();
//     }
//   }
//   fprintf(stderr, "my cilkrts_hyper_create\n");
//   real___cilkrts_hyper_create(hb);
//   enable_checking();
// }

// typedef void *(*__cilkrts_hyper_lookup_t)(__cilkrts_hyperobject_base);
// static __cilkrts_hyper_lookup_t real___cilkrts_hyper_lookup = NULL;
// CILKSAN_API
// void *__cilkrts_hyper_lookup(__cilkrts_hyperobject_base *hb) {
//   disable_checking();
//   if (real___cilkrts_hyper_lookup == NULL) {
//     real___cilkrts_hyper_lookup =
//       (__cilkrts_hyper_lookup_t)dlsym(RTLD_NEXT, "__cilkrts_hyper_lookup");
//     char *error = dlerror();
//     if (error != NULL) {
//       fputs(error, err_io);
//       fflush(err_io);
//       abort();
//     }
//   }
//   void *r = __real___cilkrts_hyper_lookup(hb);
//   enable_checking();
//   return r;
// }

// typedef cilkred_map *(*merge_reducer_maps_t)(__cilkrts_worker **,
//                                              cilkred_map *,
//                                              cilkred_map *);
// static merge_reducer_maps_t real_merge_reducer_maps = NULL;
// CILKSAN_API
// cilkred_map *merge_reducer_maps(__cilkrts_worker **w_ptr,
//                                 cilkred_map *left_map,
//                                 cilkred_map *right_map) {
//   disable_checking();
//   if (real_merge_reducer_maps == NULL) {
//     real_merge_reducer_maps =
//       (merge_reducer_maps_t)dlsym(RTLD_NEXT, "merge_reducer_maps");
//     char *error = dlerror();
//     if (error != NULL) {
//       fputs(error, err_io);
//       fflush(err_io);
//       abort();
//     }
//   }
//   std::cerr << "my merge_reducer_maps\n";
//   cilkred_map *r = real_merge_reducer_maps(w_ptr, left_map, right_map);
//   enable_checking();
//   return r;
// }
