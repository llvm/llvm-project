//===-- asan_win_dll_thunk.cc ---------------------------------------------===//
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
// This file defines a family of thunks that should be statically linked into
// the DLLs that have ASan instrumentation in order to delegate the calls to the
// shared runtime that lives in the main binary.
// See https://github.com/google/sanitizers/issues/209 for the details.
//===----------------------------------------------------------------------===//

// Only compile this code when building asan_dll_thunk.lib
// Using #ifdef rather than relying on Makefiles etc.
// simplifies the build procedure.
#ifdef ASAN_DLL_THUNK
#include "asan_init_version.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_platform_interceptors.h"

// ---------- Function interception helper functions and macros ----------- {{{1
extern "C" {
void *__stdcall GetModuleHandleA(const char *module_name);
void *__stdcall GetProcAddress(void *module, const char *proc_name);
void abort();
}

using namespace __sanitizer;

static uptr getRealProcAddressOrDie(const char *name) {
  uptr ret =
      __interception::InternalGetProcAddress((void *)GetModuleHandleA(0), name);
  if (!ret)
    abort();
  return ret;
}

// We need to intercept some functions (e.g. ASan interface, memory allocator --
// let's call them "hooks") exported by the DLL thunk and forward the hooks to
// the runtime in the main module.
// However, we don't want to keep two lists of these hooks.
// To avoid that, the list of hooks should be defined using the
// INTERCEPT_WHEN_POSSIBLE macro. Then, all these hooks can be intercepted
// at once by calling INTERCEPT_HOOKS().

// Use macro+template magic to automatically generate the list of hooks.
// Each hook at line LINE defines a template class with a static
// FunctionInterceptor<LINE>::Execute() method intercepting the hook.
// The default implementation of FunctionInterceptor<LINE> is to call
// the Execute() method corresponding to the previous line.
template<int LINE>
struct FunctionInterceptor {
  static void Execute() { FunctionInterceptor<LINE-1>::Execute(); }
};

// There shouldn't be any hooks with negative definition line number.
template<>
struct FunctionInterceptor<0> {
  static void Execute() {}
};

#define INTERCEPT_WHEN_POSSIBLE(main_function, dll_function)                   \
  template <> struct FunctionInterceptor<__LINE__> {                           \
    static void Execute() {                                                    \
      uptr wrapper = getRealProcAddressOrDie(main_function);                   \
      if (!__interception::OverrideFunction((uptr)dll_function, wrapper, 0))   \
        abort();                                                               \
      FunctionInterceptor<__LINE__ - 1>::Execute();                            \
    }                                                                          \
  };

// Special case of hooks -- ASan own interface functions.  Those are only called
// after __asan_init, thus an empty implementation is sufficient.
#define INTERFACE_FUNCTION(name)                                               \
  extern "C" __declspec(noinline) void name() {                                \
    volatile int prevent_icf = (__LINE__ << 8); (void)prevent_icf;             \
    __debugbreak();                                                            \
  }                                                                            \
  INTERCEPT_WHEN_POSSIBLE(#name, name)

// INTERCEPT_HOOKS must be used after the last INTERCEPT_WHEN_POSSIBLE.
#define INTERCEPT_HOOKS FunctionInterceptor<__LINE__>::Execute

// We can't define our own version of strlen etc. because that would lead to
// link-time or even type mismatch errors.  Instead, we can declare a function
// just to be able to get its address.  Me may miss the first few calls to the
// functions since it can be called before __asan_init, but that would lead to
// false negatives in the startup code before user's global initializers, which
// isn't a big deal.
#define INTERCEPT_LIBRARY_FUNCTION(name)                                       \
  extern "C" void name();                                                      \
  INTERCEPT_WHEN_POSSIBLE(WRAPPER_NAME(name), name)

// Disable compiler warnings that show up if we declare our own version
// of a compiler intrinsic (e.g. strlen).
#pragma warning(disable: 4391)
#pragma warning(disable: 4392)

static void InterceptHooks();
// }}}

// ---------- Function wrapping helpers ----------------------------------- {{{1
#define WRAP_V_V(name)                                                         \
  extern "C" void name() {                                                     \
    typedef void (*fntype)();                                                  \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    fn();                                                                      \
  }                                                                            \
  INTERCEPT_WHEN_POSSIBLE(#name, name);

#define WRAP_V_W(name)                                                         \
  extern "C" void name(void *arg) {                                            \
    typedef void (*fntype)(void *arg);                                         \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    fn(arg);                                                                   \
  }                                                                            \
  INTERCEPT_WHEN_POSSIBLE(#name, name);

#define WRAP_V_WW(name)                                                        \
  extern "C" void name(void *arg1, void *arg2) {                               \
    typedef void (*fntype)(void *, void *);                                    \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    fn(arg1, arg2);                                                            \
  }                                                                            \
  INTERCEPT_WHEN_POSSIBLE(#name, name);

#define WRAP_V_WWW(name)                                                       \
  extern "C" void name(void *arg1, void *arg2, void *arg3) {                   \
    typedef void *(*fntype)(void *, void *, void *);                           \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    fn(arg1, arg2, arg3);                                                      \
  }                                                                            \
  INTERCEPT_WHEN_POSSIBLE(#name, name);

#define WRAP_W_V(name)                                                         \
  extern "C" void *name() {                                                    \
    typedef void *(*fntype)();                                                 \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    return fn();                                                               \
  }                                                                            \
  INTERCEPT_WHEN_POSSIBLE(#name, name);

#define WRAP_W_W(name)                                                         \
  extern "C" void *name(void *arg) {                                           \
    typedef void *(*fntype)(void *arg);                                        \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    return fn(arg);                                                            \
  }                                                                            \
  INTERCEPT_WHEN_POSSIBLE(#name, name);

#define WRAP_W_WW(name)                                                        \
  extern "C" void *name(void *arg1, void *arg2) {                              \
    typedef void *(*fntype)(void *, void *);                                   \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    return fn(arg1, arg2);                                                     \
  }                                                                            \
  INTERCEPT_WHEN_POSSIBLE(#name, name);

#define WRAP_W_WWW(name)                                                       \
  extern "C" void *name(void *arg1, void *arg2, void *arg3) {                  \
    typedef void *(*fntype)(void *, void *, void *);                           \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    return fn(arg1, arg2, arg3);                                               \
  }                                                                            \
  INTERCEPT_WHEN_POSSIBLE(#name, name);

#define WRAP_W_WWWW(name)                                                      \
  extern "C" void *name(void *arg1, void *arg2, void *arg3, void *arg4) {      \
    typedef void *(*fntype)(void *, void *, void *, void *);                   \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    return fn(arg1, arg2, arg3, arg4);                                         \
  }                                                                            \
  INTERCEPT_WHEN_POSSIBLE(#name, name);

#define WRAP_W_WWWWW(name)                                                     \
  extern "C" void *name(void *arg1, void *arg2, void *arg3, void *arg4,        \
                        void *arg5) {                                          \
    typedef void *(*fntype)(void *, void *, void *, void *, void *);           \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    return fn(arg1, arg2, arg3, arg4, arg5);                                   \
  }                                                                            \
  INTERCEPT_WHEN_POSSIBLE(#name, name);

#define WRAP_W_WWWWWW(name)                                                    \
  extern "C" void *name(void *arg1, void *arg2, void *arg3, void *arg4,        \
                        void *arg5, void *arg6) {                              \
    typedef void *(*fntype)(void *, void *, void *, void *, void *, void *);   \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    return fn(arg1, arg2, arg3, arg4, arg5, arg6);                             \
  }                                                                            \
  INTERCEPT_WHEN_POSSIBLE(#name, name);
// }}}

// ----------------- ASan own interface functions --------------------
// Don't use the INTERFACE_FUNCTION machinery for this function as we actually
// want to call it in the __asan_init interceptor.
WRAP_W_V(__asan_should_detect_stack_use_after_return)

extern "C" {
  int __asan_option_detect_stack_use_after_return;

  // Manually wrap __asan_init as we need to initialize
  // __asan_option_detect_stack_use_after_return afterwards.
  void __asan_init() {
    typedef void (*fntype)();
    static fntype fn = 0;
    // __asan_init is expected to be called by only one thread.
    if (fn) return;

    fn = (fntype)getRealProcAddressOrDie("__asan_init");
    fn();
    __asan_option_detect_stack_use_after_return =
        (__asan_should_detect_stack_use_after_return() != 0);

    InterceptHooks();
  }
}

extern "C" void __asan_version_mismatch_check() {
  // Do nothing.
}

INTERFACE_FUNCTION(__asan_handle_no_return)

INTERFACE_FUNCTION(__asan_report_store1)
INTERFACE_FUNCTION(__asan_report_store2)
INTERFACE_FUNCTION(__asan_report_store4)
INTERFACE_FUNCTION(__asan_report_store8)
INTERFACE_FUNCTION(__asan_report_store16)
INTERFACE_FUNCTION(__asan_report_store_n)

INTERFACE_FUNCTION(__asan_report_load1)
INTERFACE_FUNCTION(__asan_report_load2)
INTERFACE_FUNCTION(__asan_report_load4)
INTERFACE_FUNCTION(__asan_report_load8)
INTERFACE_FUNCTION(__asan_report_load16)
INTERFACE_FUNCTION(__asan_report_load_n)

INTERFACE_FUNCTION(__asan_store1)
INTERFACE_FUNCTION(__asan_store2)
INTERFACE_FUNCTION(__asan_store4)
INTERFACE_FUNCTION(__asan_store8)
INTERFACE_FUNCTION(__asan_store16)
INTERFACE_FUNCTION(__asan_storeN)

INTERFACE_FUNCTION(__asan_load1)
INTERFACE_FUNCTION(__asan_load2)
INTERFACE_FUNCTION(__asan_load4)
INTERFACE_FUNCTION(__asan_load8)
INTERFACE_FUNCTION(__asan_load16)
INTERFACE_FUNCTION(__asan_loadN)

INTERFACE_FUNCTION(__asan_memcpy);
INTERFACE_FUNCTION(__asan_memset);
INTERFACE_FUNCTION(__asan_memmove);

INTERFACE_FUNCTION(__asan_set_shadow_00);
INTERFACE_FUNCTION(__asan_set_shadow_f1);
INTERFACE_FUNCTION(__asan_set_shadow_f2);
INTERFACE_FUNCTION(__asan_set_shadow_f3);
INTERFACE_FUNCTION(__asan_set_shadow_f5);
INTERFACE_FUNCTION(__asan_set_shadow_f8);

INTERFACE_FUNCTION(__asan_alloca_poison);
INTERFACE_FUNCTION(__asan_allocas_unpoison);

INTERFACE_FUNCTION(__asan_register_globals)
INTERFACE_FUNCTION(__asan_unregister_globals)

INTERFACE_FUNCTION(__asan_before_dynamic_init)
INTERFACE_FUNCTION(__asan_after_dynamic_init)

INTERFACE_FUNCTION(__asan_poison_stack_memory)
INTERFACE_FUNCTION(__asan_unpoison_stack_memory)

INTERFACE_FUNCTION(__asan_poison_memory_region)
INTERFACE_FUNCTION(__asan_unpoison_memory_region)

INTERFACE_FUNCTION(__asan_address_is_poisoned)
INTERFACE_FUNCTION(__asan_region_is_poisoned)

INTERFACE_FUNCTION(__asan_get_current_fake_stack)
INTERFACE_FUNCTION(__asan_addr_is_in_fake_stack)

INTERFACE_FUNCTION(__asan_stack_malloc_0)
INTERFACE_FUNCTION(__asan_stack_malloc_1)
INTERFACE_FUNCTION(__asan_stack_malloc_2)
INTERFACE_FUNCTION(__asan_stack_malloc_3)
INTERFACE_FUNCTION(__asan_stack_malloc_4)
INTERFACE_FUNCTION(__asan_stack_malloc_5)
INTERFACE_FUNCTION(__asan_stack_malloc_6)
INTERFACE_FUNCTION(__asan_stack_malloc_7)
INTERFACE_FUNCTION(__asan_stack_malloc_8)
INTERFACE_FUNCTION(__asan_stack_malloc_9)
INTERFACE_FUNCTION(__asan_stack_malloc_10)

INTERFACE_FUNCTION(__asan_stack_free_0)
INTERFACE_FUNCTION(__asan_stack_free_1)
INTERFACE_FUNCTION(__asan_stack_free_2)
INTERFACE_FUNCTION(__asan_stack_free_4)
INTERFACE_FUNCTION(__asan_stack_free_5)
INTERFACE_FUNCTION(__asan_stack_free_6)
INTERFACE_FUNCTION(__asan_stack_free_7)
INTERFACE_FUNCTION(__asan_stack_free_8)
INTERFACE_FUNCTION(__asan_stack_free_9)
INTERFACE_FUNCTION(__asan_stack_free_10)

// FIXME: we might want to have a sanitizer_win_dll_thunk?
INTERFACE_FUNCTION(__sanitizer_annotate_contiguous_container)
INTERFACE_FUNCTION(__sanitizer_contiguous_container_find_bad_address)
INTERFACE_FUNCTION(__sanitizer_cov)
INTERFACE_FUNCTION(__sanitizer_cov_dump)
INTERFACE_FUNCTION(__sanitizer_cov_indir_call16)
INTERFACE_FUNCTION(__sanitizer_cov_init)
INTERFACE_FUNCTION(__sanitizer_cov_module_init)
INTERFACE_FUNCTION(__sanitizer_cov_trace_basic_block)
INTERFACE_FUNCTION(__sanitizer_cov_trace_func_enter)
INTERFACE_FUNCTION(__sanitizer_cov_with_check)
INTERFACE_FUNCTION(__sanitizer_get_allocated_size)
INTERFACE_FUNCTION(__sanitizer_get_coverage_guards)
INTERFACE_FUNCTION(__sanitizer_get_coverage_pc_buffer_pos)
INTERFACE_FUNCTION(__sanitizer_get_current_allocated_bytes)
INTERFACE_FUNCTION(__sanitizer_get_estimated_allocated_size)
INTERFACE_FUNCTION(__sanitizer_get_free_bytes)
INTERFACE_FUNCTION(__sanitizer_get_heap_size)
INTERFACE_FUNCTION(__sanitizer_get_ownership)
INTERFACE_FUNCTION(__sanitizer_get_total_unique_caller_callee_pairs)
INTERFACE_FUNCTION(__sanitizer_get_total_unique_coverage)
INTERFACE_FUNCTION(__sanitizer_get_unmapped_bytes)
INTERFACE_FUNCTION(__sanitizer_maybe_open_cov_file)
INTERFACE_FUNCTION(__sanitizer_print_stack_trace)
INTERFACE_FUNCTION(__sanitizer_symbolize_pc)
INTERFACE_FUNCTION(__sanitizer_symbolize_data)
INTERFACE_FUNCTION(__sanitizer_ptr_cmp)
INTERFACE_FUNCTION(__sanitizer_ptr_sub)
INTERFACE_FUNCTION(__sanitizer_report_error_summary)
INTERFACE_FUNCTION(__sanitizer_reset_coverage)
INTERFACE_FUNCTION(__sanitizer_get_number_of_counters)
INTERFACE_FUNCTION(__sanitizer_update_counter_bitset_and_clear_counters)
INTERFACE_FUNCTION(__sanitizer_sandbox_on_notify)
INTERFACE_FUNCTION(__sanitizer_set_coverage_pc_buffer)
INTERFACE_FUNCTION(__sanitizer_set_death_callback)
INTERFACE_FUNCTION(__sanitizer_set_report_path)
INTERFACE_FUNCTION(__sanitizer_set_report_fd)
INTERFACE_FUNCTION(__sanitizer_unaligned_load16)
INTERFACE_FUNCTION(__sanitizer_unaligned_load32)
INTERFACE_FUNCTION(__sanitizer_unaligned_load64)
INTERFACE_FUNCTION(__sanitizer_unaligned_store16)
INTERFACE_FUNCTION(__sanitizer_unaligned_store32)
INTERFACE_FUNCTION(__sanitizer_unaligned_store64)
INTERFACE_FUNCTION(__sanitizer_verify_contiguous_container)
INTERFACE_FUNCTION(__sanitizer_install_malloc_and_free_hooks)
INTERFACE_FUNCTION(__sanitizer_start_switch_fiber)
INTERFACE_FUNCTION(__sanitizer_finish_switch_fiber)

// TODO(timurrrr): Add more interface functions on the as-needed basis.

// ----------------- Memory allocation functions ---------------------
WRAP_V_W(free)
WRAP_V_W(_free_base)
WRAP_V_WW(_free_dbg)

WRAP_W_W(malloc)
WRAP_W_W(_malloc_base)
WRAP_W_WWWW(_malloc_dbg)

WRAP_W_WW(calloc)
WRAP_W_WW(_calloc_base)
WRAP_W_WWWWW(_calloc_dbg)
WRAP_W_WWW(_calloc_impl)

WRAP_W_WW(realloc)
WRAP_W_WW(_realloc_base)
WRAP_W_WWW(_realloc_dbg)
WRAP_W_WWW(_recalloc)

WRAP_W_W(_msize)
WRAP_W_W(_expand)
WRAP_W_W(_expand_dbg)

// TODO(timurrrr): Might want to add support for _aligned_* allocation
// functions to detect a bit more bugs.  Those functions seem to wrap malloc().

// TODO(timurrrr): Do we need to add _Crt* stuff here? (see asan_malloc_win.cc).

INTERCEPT_LIBRARY_FUNCTION(atoi);
INTERCEPT_LIBRARY_FUNCTION(atol);

#ifdef _WIN64
INTERCEPT_LIBRARY_FUNCTION(__C_specific_handler);
#else
INTERCEPT_LIBRARY_FUNCTION(_except_handler3);

// _except_handler4 checks -GS cookie which is different for each module, so we
// can't use INTERCEPT_LIBRARY_FUNCTION(_except_handler4).
INTERCEPTOR(int, _except_handler4, void *a, void *b, void *c, void *d) {
  __asan_handle_no_return();
  return REAL(_except_handler4)(a, b, c, d);
}
#endif

INTERCEPT_LIBRARY_FUNCTION(frexp);
INTERCEPT_LIBRARY_FUNCTION(longjmp);
#if SANITIZER_INTERCEPT_MEMCHR
INTERCEPT_LIBRARY_FUNCTION(memchr);
#endif
INTERCEPT_LIBRARY_FUNCTION(memcmp);
INTERCEPT_LIBRARY_FUNCTION(memcpy);
INTERCEPT_LIBRARY_FUNCTION(memmove);
INTERCEPT_LIBRARY_FUNCTION(memset);
INTERCEPT_LIBRARY_FUNCTION(strcat);  // NOLINT
INTERCEPT_LIBRARY_FUNCTION(strchr);
INTERCEPT_LIBRARY_FUNCTION(strcmp);
INTERCEPT_LIBRARY_FUNCTION(strcpy);  // NOLINT
INTERCEPT_LIBRARY_FUNCTION(strcspn);
INTERCEPT_LIBRARY_FUNCTION(strdup);
INTERCEPT_LIBRARY_FUNCTION(strlen);
INTERCEPT_LIBRARY_FUNCTION(strncat);
INTERCEPT_LIBRARY_FUNCTION(strncmp);
INTERCEPT_LIBRARY_FUNCTION(strncpy);
INTERCEPT_LIBRARY_FUNCTION(strnlen);
INTERCEPT_LIBRARY_FUNCTION(strpbrk);
INTERCEPT_LIBRARY_FUNCTION(strrchr);
INTERCEPT_LIBRARY_FUNCTION(strspn);
INTERCEPT_LIBRARY_FUNCTION(strstr);
INTERCEPT_LIBRARY_FUNCTION(strtol);
INTERCEPT_LIBRARY_FUNCTION(wcslen);

// Must be after all the interceptor declarations due to the way INTERCEPT_HOOKS
// is defined.
void InterceptHooks() {
  INTERCEPT_HOOKS();
#ifndef _WIN64
  INTERCEPT_FUNCTION(_except_handler4);
#endif
}

// We want to call __asan_init before C/C++ initializers/constructors are
// executed, otherwise functions like memset might be invoked.
// For some strange reason, merely linking in asan_preinit.cc doesn't work
// as the callback is never called...  Is link.exe doing something too smart?

// In DLLs, the callbacks are expected to return 0,
// otherwise CRT initialization fails.
static int call_asan_init() {
  __asan_init();
  return 0;
}
#pragma section(".CRT$XIB", long, read)  // NOLINT
__declspec(allocate(".CRT$XIB")) int (*__asan_preinit)() = call_asan_init;

#endif // ASAN_DLL_THUNK
