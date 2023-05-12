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

#include "hwasan.h"
#include "hwasan_checks.h"
#include "hwasan_thread.h"
#include "hwasan_thread_list.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_linux.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

#if !SANITIZER_FUCHSIA

using namespace __hwasan;

#  if HWASAN_WITH_INTERCEPTORS

#    define COMMON_SYSCALL_PRE_READ_RANGE(p, s) __hwasan_loadN((uptr)p, (uptr)s)
#    define COMMON_SYSCALL_PRE_WRITE_RANGE(p, s) \
      __hwasan_storeN((uptr)p, (uptr)s)
#    define COMMON_SYSCALL_POST_READ_RANGE(p, s) \
      do {                                       \
        (void)(p);                               \
        (void)(s);                               \
      } while (false)
#    define COMMON_SYSCALL_POST_WRITE_RANGE(p, s) \
      do {                                        \
        (void)(p);                                \
        (void)(s);                                \
      } while (false)
#    include "sanitizer_common/sanitizer_common_syscalls.inc"
#    include "sanitizer_common/sanitizer_syscalls_netbsd.inc"

struct ThreadStartArg {
  __sanitizer_sigset_t starting_sigset_;
};

static void *HwasanThreadStartFunc(void *arg) {
  __hwasan_thread_enter();
  ThreadStartArg A = *reinterpret_cast<ThreadStartArg *>(arg);
  SetSigProcMask(&A.starting_sigset_, nullptr);
  InternalFree(arg);
  auto self = GetThreadSelf();
  auto args = hwasanThreadArgRetval().GetArgs(self);
  void *retval = (*args.routine)(args.arg_retval);
  hwasanThreadArgRetval().Finish(self, retval);
  return retval;
}

extern "C" {
int pthread_attr_getdetachstate(void *attr, int *v);
}

INTERCEPTOR(int, pthread_create, void *thread, void *attr,
            void *(*callback)(void *), void *param) {
  EnsureMainThreadIDIsCorrect();
  ScopedTaggingDisabler tagging_disabler;
  int detached = 0;
  if (attr)
    pthread_attr_getdetachstate(attr, &detached);
  ThreadStartArg *A = (ThreadStartArg *)InternalAlloc(sizeof(ThreadStartArg));
  ScopedBlockSignals block(&A->starting_sigset_);
  // ASAN uses the same approach to disable leaks from pthread_create.
#    if CAN_SANITIZE_LEAKS
  __lsan::ScopedInterceptorDisabler lsan_disabler;
#    endif

  int result;
  hwasanThreadArgRetval().Create(detached, {callback, param}, [&]() -> uptr {
    result = REAL(pthread_create)(thread, attr, &HwasanThreadStartFunc, A);
    return result ? 0 : *(uptr *)(thread);
  });
  if (result != 0)
    InternalFree(A);
  return result;
}

INTERCEPTOR(int, pthread_join, void *thread, void **retval) {
  int result;
  hwasanThreadArgRetval().Join((uptr)thread, [&]() {
    result = REAL(pthread_join)(thread, retval);
    return !result;
  });
  return result;
}

INTERCEPTOR(int, pthread_detach, void *thread) {
  int result;
  hwasanThreadArgRetval().Detach((uptr)thread, [&]() {
    result = REAL(pthread_detach)(thread);
    return !result;
  });
  return result;
}

INTERCEPTOR(int, pthread_exit, void *retval) {
  auto *t = GetCurrentThread();
  if (t && !t->IsMainThread())
    hwasanThreadArgRetval().Finish(GetThreadSelf(), retval);
  return REAL(pthread_exit)(retval);
}

#    if SANITIZER_GLIBC
INTERCEPTOR(int, pthread_tryjoin_np, void *thread, void **ret) {
  int result;
  hwasanThreadArgRetval().Join((uptr)thread, [&]() {
    result = REAL(pthread_tryjoin_np)(thread, ret);
    return !result;
  });
  return result;
}

INTERCEPTOR(int, pthread_timedjoin_np, void *thread, void **ret,
            const struct timespec *abstime) {
  int result;
  hwasanThreadArgRetval().Join((uptr)thread, [&]() {
    result = REAL(pthread_timedjoin_np)(thread, ret, abstime);
    return !result;
  });
  return result;
}
#    endif

DEFINE_REAL_PTHREAD_FUNCTIONS

DEFINE_REAL(int, vfork)
DECLARE_EXTERN_INTERCEPTOR_AND_WRAPPER(int, vfork)

// Get and/or change the set of blocked signals.
extern "C" int sigprocmask(int __how, const __hw_sigset_t *__restrict __set,
                           __hw_sigset_t *__restrict __oset);
#    define SIG_BLOCK 0
#    define SIG_SETMASK 2
extern "C" int __sigjmp_save(__hw_sigjmp_buf env, int savemask) {
  env[0].__magic = kHwJmpBufMagic;
  env[0].__mask_was_saved =
      (savemask &&
       sigprocmask(SIG_BLOCK, (__hw_sigset_t *)0, &env[0].__saved_mask) == 0);
  return 0;
}

static void __attribute__((always_inline))
InternalLongjmp(__hw_register_buf env, int retval) {
#    if defined(__aarch64__)
  constexpr size_t kSpIndex = 13;
#    elif defined(__x86_64__)
  constexpr size_t kSpIndex = 6;
#    elif SANITIZER_RISCV64
  constexpr size_t kSpIndex = 13;
#    endif

  // Clear all memory tags on the stack between here and where we're going.
  unsigned long long stack_pointer = env[kSpIndex];
  // The stack pointer should never be tagged, so we don't need to clear the
  // tag for this function call.
  __hwasan_handle_longjmp((void *)stack_pointer);

  // Run code for handling a longjmp.
  // Need to use a register that isn't going to be loaded from the environment
  // buffer -- hence why we need to specify the register to use.
  // Must implement this ourselves, since we don't know the order of registers
  // in different libc implementations and many implementations mangle the
  // stack pointer so we can't use it without knowing the demangling scheme.
#    if defined(__aarch64__)
  register long int retval_tmp asm("x1") = retval;
  register void *env_address asm("x0") = &env[0];
  asm volatile(
      "ldp	x19, x20, [%0, #0<<3];"
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
#    elif defined(__x86_64__)
  register long int retval_tmp asm("%rsi") = retval;
  register void *env_address asm("%rdi") = &env[0];
  asm volatile(
      // Restore registers.
      "mov (0*8)(%0),%%rbx;"
      "mov (1*8)(%0),%%rbp;"
      "mov (2*8)(%0),%%r12;"
      "mov (3*8)(%0),%%r13;"
      "mov (4*8)(%0),%%r14;"
      "mov (5*8)(%0),%%r15;"
      "mov (6*8)(%0),%%rsp;"
      "mov (7*8)(%0),%%rdx;"
      // Return 1 if retval is 0.
      "mov $1,%%rax;"
      "test %1,%1;"
      "cmovnz %1,%%rax;"
      "jmp *%%rdx;" ::"r"(env_address),
      "r"(retval_tmp));
#    elif SANITIZER_RISCV64
  register long int retval_tmp asm("x11") = retval;
  register void *env_address asm("x10") = &env[0];
  asm volatile(
      "ld     ra,   0<<3(%0);"
      "ld     s0,   1<<3(%0);"
      "ld     s1,   2<<3(%0);"
      "ld     s2,   3<<3(%0);"
      "ld     s3,   4<<3(%0);"
      "ld     s4,   5<<3(%0);"
      "ld     s5,   6<<3(%0);"
      "ld     s6,   7<<3(%0);"
      "ld     s7,   8<<3(%0);"
      "ld     s8,   9<<3(%0);"
      "ld     s9,   10<<3(%0);"
      "ld     s10,  11<<3(%0);"
      "ld     s11,  12<<3(%0);"
#      if __riscv_float_abi_double
      "fld    fs0,  14<<3(%0);"
      "fld    fs1,  15<<3(%0);"
      "fld    fs2,  16<<3(%0);"
      "fld    fs3,  17<<3(%0);"
      "fld    fs4,  18<<3(%0);"
      "fld    fs5,  19<<3(%0);"
      "fld    fs6,  20<<3(%0);"
      "fld    fs7,  21<<3(%0);"
      "fld    fs8,  22<<3(%0);"
      "fld    fs9,  23<<3(%0);"
      "fld    fs10, 24<<3(%0);"
      "fld    fs11, 25<<3(%0);"
#      elif __riscv_float_abi_soft
#      else
#        error "Unsupported case"
#      endif
      "ld     a4, 13<<3(%0);"
      "mv     sp, a4;"
      // Return the value requested to return through arguments.
      // This should be in x11 given what we requested above.
      "seqz   a0, %1;"
      "add    a0, a0, %1;"
      "ret;"
      : "+r"(env_address)
      : "r"(retval_tmp));
#    endif
}

INTERCEPTOR(void, siglongjmp, __hw_sigjmp_buf env, int val) {
  if (env[0].__magic != kHwJmpBufMagic) {
    Printf(
        "WARNING: Unexpected bad jmp_buf. Either setjmp was not called or "
        "there is a bug in HWASan.\n");
    return REAL(siglongjmp)(env, val);
  }

  if (env[0].__mask_was_saved)
    // Restore the saved signal mask.
    (void)sigprocmask(SIG_SETMASK, &env[0].__saved_mask, (__hw_sigset_t *)0);
  InternalLongjmp(env[0].__jmpbuf, val);
}

// Required since glibc libpthread calls __libc_longjmp on pthread_exit, and
// _setjmp on start_thread.  Hence we have to intercept the longjmp on
// pthread_exit so the __hw_jmp_buf order matches.
INTERCEPTOR(void, __libc_longjmp, __hw_jmp_buf env, int val) {
  if (env[0].__magic != kHwJmpBufMagic)
    return REAL(__libc_longjmp)(env, val);
  InternalLongjmp(env[0].__jmpbuf, val);
}

INTERCEPTOR(void, longjmp, __hw_jmp_buf env, int val) {
  if (env[0].__magic != kHwJmpBufMagic) {
    Printf(
        "WARNING: Unexpected bad jmp_buf. Either setjmp was not called or "
        "there is a bug in HWASan.\n");
    return REAL(longjmp)(env, val);
  }
  InternalLongjmp(env[0].__jmpbuf, val);
}
#    undef SIG_BLOCK
#    undef SIG_SETMASK

#  endif  // HWASAN_WITH_INTERCEPTORS

namespace __hwasan {

int OnExit() {
  if (CAN_SANITIZE_LEAKS && common_flags()->detect_leaks &&
      __lsan::HasReportedLeaks()) {
    return common_flags()->exitcode;
  }
  // FIXME: ask frontend whether we need to return failure.
  return 0;
}

}  // namespace __hwasan

namespace __hwasan {

void InitializeInterceptors() {
  static int inited = 0;
  CHECK_EQ(inited, 0);

#  if HWASAN_WITH_INTERCEPTORS
#    if defined(__linux__)
  INTERCEPT_FUNCTION(__libc_longjmp);
  INTERCEPT_FUNCTION(longjmp);
  INTERCEPT_FUNCTION(siglongjmp);
  INTERCEPT_FUNCTION(vfork);
#    endif  // __linux__
  INTERCEPT_FUNCTION(pthread_create);
  INTERCEPT_FUNCTION(pthread_join);
  INTERCEPT_FUNCTION(pthread_detach);
  INTERCEPT_FUNCTION(pthread_exit);
#    if SANITIZER_GLIBC
  INTERCEPT_FUNCTION(pthread_tryjoin_np);
  INTERCEPT_FUNCTION(pthread_timedjoin_np);
#    endif
#  endif

  inited = 1;
}
}  // namespace __hwasan

#endif  // #if !SANITIZER_FUCHSIA
