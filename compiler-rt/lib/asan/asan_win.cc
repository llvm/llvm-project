//===-- asan_win.cc -------------------------------------------------------===//
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
// Windows-specific details.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdlib.h>

#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_thread.h"
#include "asan_mapping.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"

using namespace __asan;  // NOLINT

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
int __asan_should_detect_stack_use_after_return() {
  __asan_init();
  return __asan_option_detect_stack_use_after_return;
}

// -------------------- A workaround for the abscence of weak symbols ----- {{{
// We don't have a direct equivalent of weak symbols when using MSVC, but we can
// use the /alternatename directive to tell the linker to default a specific
// symbol to a specific value, which works nicely for allocator hooks and
// __asan_default_options().
void __sanitizer_default_malloc_hook(void *ptr, uptr size) { }
void __sanitizer_default_free_hook(void *ptr) { }
const char* __asan_default_default_options() { return ""; }
const char* __asan_default_default_suppressions() { return ""; }
void __asan_default_on_error() {}
// 64-bit msvc will not prepend an underscore for symbols.
#ifdef _WIN64
#pragma comment(linker, "/alternatename:__sanitizer_malloc_hook=__sanitizer_default_malloc_hook")  // NOLINT
#pragma comment(linker, "/alternatename:__sanitizer_free_hook=__sanitizer_default_free_hook")      // NOLINT
#pragma comment(linker, "/alternatename:__asan_default_options=__asan_default_default_options")    // NOLINT
#pragma comment(linker, "/alternatename:__asan_default_suppressions=__asan_default_default_suppressions")    // NOLINT
#pragma comment(linker, "/alternatename:__asan_on_error=__asan_default_on_error")                  // NOLINT
#else
#pragma comment(linker, "/alternatename:___sanitizer_malloc_hook=___sanitizer_default_malloc_hook")  // NOLINT
#pragma comment(linker, "/alternatename:___sanitizer_free_hook=___sanitizer_default_free_hook")      // NOLINT
#pragma comment(linker, "/alternatename:___asan_default_options=___asan_default_default_options")    // NOLINT
#pragma comment(linker, "/alternatename:___asan_default_suppressions=___asan_default_default_suppressions")    // NOLINT
#pragma comment(linker, "/alternatename:___asan_on_error=___asan_default_on_error")                  // NOLINT
#endif
// }}}
}  // extern "C"

// ---------------------- Windows-specific inteceptors ---------------- {{{
INTERCEPTOR_WINAPI(void, RaiseException, void *a, void *b, void *c, void *d) {
  CHECK(REAL(RaiseException));
  __asan_handle_no_return();
  REAL(RaiseException)(a, b, c, d);
}


#ifdef _WIN64

INTERCEPTOR_WINAPI(int, __C_specific_handler, void *a, void *b, void *c, void *d) {  // NOLINT
  CHECK(REAL(__C_specific_handler));
  __asan_handle_no_return();
  return REAL(__C_specific_handler)(a, b, c, d);
}

#else

INTERCEPTOR(int, _except_handler3, void *a, void *b, void *c, void *d) {
  CHECK(REAL(_except_handler3));
  __asan_handle_no_return();
  return REAL(_except_handler3)(a, b, c, d);
}

#if ASAN_DYNAMIC
// This handler is named differently in -MT and -MD CRTs.
#define _except_handler4 _except_handler4_common
#endif
INTERCEPTOR(int, _except_handler4, void *a, void *b, void *c, void *d) {
  CHECK(REAL(_except_handler4));
  __asan_handle_no_return();
  return REAL(_except_handler4)(a, b, c, d);
}
#endif

static thread_return_t THREAD_CALLING_CONV asan_thread_start(void *arg) {
  AsanThread *t = (AsanThread*)arg;
  SetCurrentThread(t);
  return t->ThreadStart(GetTid(), /* signal_thread_is_registered */ nullptr);
}

INTERCEPTOR_WINAPI(DWORD, CreateThread,
                   void* security, uptr stack_size,
                   DWORD (__stdcall *start_routine)(void*), void* arg,
                   DWORD thr_flags, void* tid) {
  // Strict init-order checking is thread-hostile.
  if (flags()->strict_init_order)
    StopInitOrderChecking();
  GET_STACK_TRACE_THREAD;
  // FIXME: The CreateThread interceptor is not the same as a pthread_create
  // one.  This is a bandaid fix for PR22025.
  bool detached = false;  // FIXME: how can we determine it on Windows?
  u32 current_tid = GetCurrentTidOrInvalid();
  AsanThread *t =
        AsanThread::Create(start_routine, arg, current_tid, &stack, detached);
  return REAL(CreateThread)(security, stack_size,
                            asan_thread_start, t, thr_flags, tid);
}

namespace {
BlockingMutex mu_for_thread_tracking(LINKER_INITIALIZED);

void EnsureWorkerThreadRegistered() {
  // FIXME: GetCurrentThread relies on TSD, which might not play well with
  // system thread pools.  We might want to use something like reference
  // counting to zero out GetCurrentThread() underlying storage when the last
  // work item finishes?  Or can we disable reclaiming of threads in the pool?
  BlockingMutexLock l(&mu_for_thread_tracking);
  if (__asan::GetCurrentThread())
    return;

  AsanThread *t = AsanThread::Create(
      /* start_routine */ nullptr, /* arg */ nullptr,
      /* parent_tid */ -1, /* stack */ nullptr, /* detached */ true);
  t->Init();
  asanThreadRegistry().StartThread(t->tid(), 0, 0);
  SetCurrentThread(t);
}
}  // namespace

INTERCEPTOR_WINAPI(DWORD, NtWaitForWorkViaWorkerFactory, DWORD a, DWORD b) {
  // NtWaitForWorkViaWorkerFactory is called from system worker pool threads to
  // query work scheduled by BindIoCompletionCallback, QueueUserWorkItem, etc.
  // System worker pool threads are created at arbitraty point in time and
  // without using CreateThread, so we wrap NtWaitForWorkViaWorkerFactory
  // instead and don't register a specific parent_tid/stack.
  EnsureWorkerThreadRegistered();
  return REAL(NtWaitForWorkViaWorkerFactory)(a, b);
}

// }}}

namespace __asan {

void InitializePlatformInterceptors() {
  ASAN_INTERCEPT_FUNC(CreateThread);
  ASAN_INTERCEPT_FUNC(RaiseException);

#ifdef _WIN64
  ASAN_INTERCEPT_FUNC(__C_specific_handler);
#else
  ASAN_INTERCEPT_FUNC(_except_handler3);
  ASAN_INTERCEPT_FUNC(_except_handler4);
#endif

  // NtWaitForWorkViaWorkerFactory is always linked dynamically.
  CHECK(::__interception::OverrideFunction(
      "NtWaitForWorkViaWorkerFactory",
      (uptr)WRAP(NtWaitForWorkViaWorkerFactory),
      (uptr *)&REAL(NtWaitForWorkViaWorkerFactory)));
}

void AsanApplyToGlobals(globals_op_fptr op, const void *needle) {
  UNIMPLEMENTED();
}

// ---------------------- TSD ---------------- {{{
static bool tsd_key_inited = false;

static __declspec(thread) void *fake_tsd = 0;

void AsanTSDInit(void (*destructor)(void *tsd)) {
  // FIXME: we're ignoring the destructor for now.
  tsd_key_inited = true;
}

void *AsanTSDGet() {
  CHECK(tsd_key_inited);
  return fake_tsd;
}

void AsanTSDSet(void *tsd) {
  CHECK(tsd_key_inited);
  fake_tsd = tsd;
}

void PlatformTSDDtor(void *tsd) {
  AsanThread::TSDDtor(tsd);
}
// }}}

// ---------------------- Various stuff ---------------- {{{
void *AsanDoesNotSupportStaticLinkage() {
#if defined(_DEBUG)
#error Please build the runtime with a non-debug CRT: /MD or /MT
#endif
  return 0;
}

void AsanCheckDynamicRTPrereqs() {}

void AsanCheckIncompatibleRT() {}

void ReadContextStack(void *context, uptr *stack, uptr *ssize) {
  UNIMPLEMENTED();
}

void AsanOnDeadlySignal(int, void *siginfo, void *context) {
  UNIMPLEMENTED();
}

#if SANITIZER_WINDOWS64
// Exception handler for dealing with shadow memory.
static LONG CALLBACK
ShadowExceptionHandler(PEXCEPTION_POINTERS exception_pointers) {
  static uptr page_size = GetPageSizeCached();
  static uptr alloc_granularity = GetMmapGranularity();
  // Only handle access violations.
  if (exception_pointers->ExceptionRecord->ExceptionCode !=
      EXCEPTION_ACCESS_VIOLATION) {
    return EXCEPTION_CONTINUE_SEARCH;
  }

  // Only handle access violations that land within the shadow memory.
  uptr addr =
      (uptr)(exception_pointers->ExceptionRecord->ExceptionInformation[1]);

  // Check valid shadow range.
  if (!AddrIsInShadow(addr)) return EXCEPTION_CONTINUE_SEARCH;

  // This is an access violation while trying to read from the shadow. Commit
  // the relevant page and let execution continue.

  // Determine the address of the page that is being accessed.
  uptr page = RoundDownTo(addr, page_size);

  // Query the existing page.
  MEMORY_BASIC_INFORMATION mem_info = {};
  if (::VirtualQuery((LPVOID)page, &mem_info, sizeof(mem_info)) == 0)
    return EXCEPTION_CONTINUE_SEARCH;

  // Commit the page.
  uptr result =
      (uptr)::VirtualAlloc((LPVOID)page, page_size, MEM_COMMIT, PAGE_READWRITE);
  if (result != page) return EXCEPTION_CONTINUE_SEARCH;

  // The page mapping succeeded, so continue execution as usual.
  return EXCEPTION_CONTINUE_EXECUTION;
}

#endif

void InitializePlatformExceptionHandlers() {
#if SANITIZER_WINDOWS64
  // On Win64, we map memory on demand with access violation handler.
  // Install our exception handler.
  CHECK(AddVectoredExceptionHandler(TRUE, &ShadowExceptionHandler));
#endif
}

static LPTOP_LEVEL_EXCEPTION_FILTER default_seh_handler;

static long WINAPI SEHHandler(EXCEPTION_POINTERS *info) {
  EXCEPTION_RECORD *exception_record = info->ExceptionRecord;
  CONTEXT *context = info->ContextRecord;

  if (exception_record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION ||
      exception_record->ExceptionCode == EXCEPTION_IN_PAGE_ERROR) {
    const char *description =
        (exception_record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION)
            ? "access-violation"
            : "in-page-error";
    SignalContext sig = SignalContext::Create(exception_record, context);
    ReportDeadlySignal(description, sig);
  }

  // FIXME: Handle EXCEPTION_STACK_OVERFLOW here.

  return default_seh_handler(info);
}

// We want to install our own exception handler (EH) to print helpful reports
// on access violations and whatnot.  Unfortunately, the CRT initializers assume
// they are run before any user code and drop any previously-installed EHs on
// the floor, so we can't install our handler inside __asan_init.
// (See crt0dat.c in the CRT sources for the details)
//
// Things get even more complicated with the dynamic runtime, as it finishes its
// initialization before the .exe module CRT begins to initialize.
//
// For the static runtime (-MT), it's enough to put a callback to
// __asan_set_seh_filter in the last section for C initializers.
//
// For the dynamic runtime (-MD), we want link the same
// asan_dynamic_runtime_thunk.lib to all the modules, thus __asan_set_seh_filter
// will be called for each instrumented module.  This ensures that at least one
// __asan_set_seh_filter call happens after the .exe module CRT is initialized.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE
int __asan_set_seh_filter() {
  // We should only store the previous handler if it's not our own handler in
  // order to avoid loops in the EH chain.
  auto prev_seh_handler = SetUnhandledExceptionFilter(SEHHandler);
  if (prev_seh_handler != &SEHHandler)
    default_seh_handler = prev_seh_handler;
  return 0;
}

#if !ASAN_DYNAMIC
// The CRT runs initializers in this order:
// - C initializers, from XIA to XIZ
// - C++ initializers, from XCA to XCZ
// Prior to 2015, the CRT set the unhandled exception filter at priority XIY,
// near the end of C initialization. Starting in 2015, it was moved to the
// beginning of C++ initialization. We set our priority to XCAB to run
// immediately after the CRT runs. This way, our exception filter is called
// first and we can delegate to their filter if appropriate.
#pragma section(".CRT$XCAB", long, read)  // NOLINT
__declspec(allocate(".CRT$XCAB"))
    int (*__intercept_seh)() = __asan_set_seh_filter;
#endif
// }}}
}  // namespace __asan

#endif  // _WIN32
