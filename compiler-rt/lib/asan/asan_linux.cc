//===-- asan_linux.cc -----------------------------------------------------===//
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
// Linux-specific details.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_LINUX

#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_procmaps.h"

#include <sys/time.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <unwind.h>

#if !SANITIZER_ANDROID
// FIXME: where to get ucontext on Android?
#include <sys/ucontext.h>
#endif

extern "C" void* _DYNAMIC;

namespace __asan {

void MaybeReexec() {
  // No need to re-exec on Linux.
}

void *AsanDoesNotSupportStaticLinkage() {
  // This will fail to link with -static.
  return &_DYNAMIC;  // defined in link.h
}

void GetPcSpBp(void *context, uptr *pc, uptr *sp, uptr *bp) {
#if SANITIZER_ANDROID
  *pc = *sp = *bp = 0;
#elif defined(__arm__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.arm_pc;
  *bp = ucontext->uc_mcontext.arm_fp;
  *sp = ucontext->uc_mcontext.arm_sp;
# elif defined(__hppa__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.sc_iaoq[0];
  /* GCC uses %r3 whenever a frame pointer is needed.  */
  *bp = ucontext->uc_mcontext.sc_gr[3];
  *sp = ucontext->uc_mcontext.sc_gr[30];
# elif defined(__x86_64__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.gregs[REG_RIP];
  *bp = ucontext->uc_mcontext.gregs[REG_RBP];
  *sp = ucontext->uc_mcontext.gregs[REG_RSP];
# elif defined(__i386__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.gregs[REG_EIP];
  *bp = ucontext->uc_mcontext.gregs[REG_EBP];
  *sp = ucontext->uc_mcontext.gregs[REG_ESP];
# elif defined(__powerpc__) || defined(__powerpc64__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.regs->nip;
  *sp = ucontext->uc_mcontext.regs->gpr[PT_R1];
  // The powerpc{,64}-linux ABIs do not specify r31 as the frame
  // pointer, but GCC always uses r31 when we need a frame pointer.
  *bp = ucontext->uc_mcontext.regs->gpr[PT_R31];
# elif defined(__sparc__)
  ucontext_t *ucontext = (ucontext_t*)context;
  uptr *stk_ptr;
# if defined (__arch64__)
  *pc = ucontext->uc_mcontext.mc_gregs[MC_PC];
  *sp = ucontext->uc_mcontext.mc_gregs[MC_O6];
  stk_ptr = (uptr *) (*sp + 2047);
  *bp = stk_ptr[15];
# else
  *pc = ucontext->uc_mcontext.gregs[REG_PC];
  *sp = ucontext->uc_mcontext.gregs[REG_O6];
  stk_ptr = (uptr *) *sp;
  *bp = stk_ptr[15];
# endif
# elif defined(__mips__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.gregs[31];
  *bp = ucontext->uc_mcontext.gregs[30];
  *sp = ucontext->uc_mcontext.gregs[29];
#else
# error "Unsupported arch"
#endif
}

bool AsanInterceptsSignal(int signum) {
  return signum == SIGSEGV && flags()->handle_segv;
}

void AsanPlatformThreadInit() {
  // Nothing here for now.
}

#if !SANITIZER_ANDROID
void ReadContextStack(void *context, uptr *stack, uptr *ssize) {
  ucontext_t *ucp = (ucontext_t*)context;
  *stack = (uptr)ucp->uc_stack.ss_sp;
  *ssize = ucp->uc_stack.ss_size;
}
#else
void ReadContextStack(void *context, uptr *stack, uptr *ssize) {
  UNIMPLEMENTED();
}
#endif

}  // namespace __asan

#endif  // SANITIZER_LINUX
