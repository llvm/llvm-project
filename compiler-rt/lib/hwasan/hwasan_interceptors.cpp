//===-- hwasan_interceptors.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
// Interceptors for standard library functions.
//
// FIXME: move as many interceptors as possible into
// sanitizer_common/sanitizer_common_interceptors.h
//===----------------------------------------------------------------------===//

#include "interception/interception.h"
#include "hwasan.h"
#include "hwasan_thread.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

#if !SANITIZER_FUCHSIA

using namespace __hwasan;

#if HWASAN_WITH_INTERCEPTORS

struct ThreadStartArg {
  thread_callback_t callback;
  void *param;
};

static void *HwasanThreadStartFunc(void *arg) {
  __hwasan_thread_enter();
  ThreadStartArg A = *reinterpret_cast<ThreadStartArg*>(arg);
  UnmapOrDie(arg, GetPageSizeCached());
  return A.callback(A.param);
}

INTERCEPTOR(int, pthread_create, void *th, void *attr, void *(*callback)(void*),
            void * param) {
  ScopedTaggingDisabler disabler;
  ThreadStartArg *A = reinterpret_cast<ThreadStartArg *> (MmapOrDie(
      GetPageSizeCached(), "pthread_create"));
  *A = {callback, param};
  int res = REAL(pthread_create)(th, attr, &HwasanThreadStartFunc, A);
  return res;
}

DEFINE_REAL(int, vfork)
DECLARE_EXTERN_INTERCEPTOR_AND_WRAPPER(int, vfork)
#endif // HWASAN_WITH_INTERCEPTORS

#if HWASAN_WITH_INTERCEPTORS && defined(__aarch64__)
// Get and/or change the set of blocked signals.
extern "C" int sigprocmask(int __how, const __hw_sigset_t *__restrict __set,
                           __hw_sigset_t *__restrict __oset);
#define SIG_BLOCK 0
#define SIG_SETMASK 2
extern "C" int __sigjmp_save(__hw_sigjmp_buf env, int savemask) {
  env[0].__mask_was_saved =
      (savemask && sigprocmask(SIG_BLOCK, (__hw_sigset_t *)0,
                               &env[0].__saved_mask) == 0);
  return 0;
}

static void __attribute__((always_inline))
InternalLongjmp(__hw_register_buf env, int retval) {
  // Clear all memory tags on the stack between here and where we're going.
  unsigned long long stack_pointer = env[13];
  // The stack pointer should never be tagged, so we don't need to clear the
  // tag for this function call.
  __hwasan_handle_longjmp((void *)stack_pointer);

  // Run code for handling a longjmp.
  // Need to use a register that isn't going to be loaded from the environment
  // buffer -- hence why we need to specify the register to use.
  // Must implement this ourselves, since we don't know the order of registers
  // in different libc implementations and many implementations mangle the
  // stack pointer so we can't use it without knowing the demangling scheme.
  register long int retval_tmp asm("x1") = retval;
  register void *env_address asm("x0") = &env[0];
  asm volatile("ldp	x19, x20, [%0, #0<<3];"
               "ldp	x21, x22, [%0, #2<<3];"
               "ldp	x23, x24, [%0, #4<<3];"
               "ldp	x25, x26, [%0, #6<<3];"
               "ldp	x27, x28, [%0, #8<<3];"
               "ldp	x29, x30, [%0, #10<<3];"
               "ldp	 d8,  d9, [%0, #14<<3];"
               "ldp	d10, d11, [%0, #16<<3];"
               "ldp	d12, d13, [%0, #18<<3];"
               "ldp	d14, d15, [%0, #20<<3];"
               "ldr	x5, [%0, #13<<3];"
               "mov	sp, x5;"
               // Return the value requested to return through arguments.
               // This should be in x1 given what we requested above.
               "cmp	%1, #0;"
               "mov	x0, #1;"
               "csel	x0, %1, x0, ne;"
               "br	x30;"
               : "+r"(env_address)
               : "r"(retval_tmp));
}

INTERCEPTOR(void, siglongjmp, __hw_sigjmp_buf env, int val) {
  if (env[0].__mask_was_saved)
    // Restore the saved signal mask.
    (void)sigprocmask(SIG_SETMASK, &env[0].__saved_mask,
                      (__hw_sigset_t *)0);
  InternalLongjmp(env[0].__jmpbuf, val);
}

// Required since glibc libpthread calls __libc_longjmp on pthread_exit, and
// _setjmp on start_thread.  Hence we have to intercept the longjmp on
// pthread_exit so the __hw_jmp_buf order matches.
INTERCEPTOR(void, __libc_longjmp, __hw_jmp_buf env, int val) {
  InternalLongjmp(env[0].__jmpbuf, val);
}

INTERCEPTOR(void, longjmp, __hw_jmp_buf env, int val) {
  InternalLongjmp(env[0].__jmpbuf, val);
}
#undef SIG_BLOCK
#undef SIG_SETMASK

#endif // HWASAN_WITH_INTERCEPTORS && __aarch64__

static void BeforeFork() {
  StackDepotLockAll();
}

static void AfterFork() {
  StackDepotUnlockAll();
}

INTERCEPTOR(int, fork, void) {
  ENSURE_HWASAN_INITED();
  BeforeFork();
  int pid = REAL(fork)();
  AfterFork();
  return pid;
}

namespace __hwasan {

int OnExit() {
  // FIXME: ask frontend whether we need to return failure.
  return 0;
}

} // namespace __hwasan

namespace __hwasan {

void InitializeInterceptors() {
  static int inited = 0;
  CHECK_EQ(inited, 0);

  INTERCEPT_FUNCTION(fork);

#if HWASAN_WITH_INTERCEPTORS
#if defined(__linux__)
  INTERCEPT_FUNCTION(vfork);
#endif  // __linux__
  INTERCEPT_FUNCTION(pthread_create);
#endif

  inited = 1;
}
} // namespace __hwasan

#endif  // #if !SANITIZER_FUCHSIA
