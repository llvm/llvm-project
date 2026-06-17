//===-- ubsan_loop_detect.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runtime support for -fsanitize-trap-loop.
//
//===----------------------------------------------------------------------===//

#include <sanitizer/ubsan_interface.h>

#if defined(__linux__) && (defined(__i386__) || defined(__x86_64__))

#include <asm/processor-flags.h>
#include <signal.h>
#include <stdint.h>
#include <sys/time.h>
#include <sys/ucontext.h>

int __ubsan_is_trap_loop(void *c) {
  auto *uc = reinterpret_cast<ucontext_t *>(c);
#if defined(__x86_64__)
  auto *ip = reinterpret_cast<const uint8_t *>(uc->uc_mcontext.gregs[REG_RIP]);
#else
  auto *ip = reinterpret_cast<const uint8_t *>(uc->uc_mcontext.gregs[REG_EIP]);
#endif
  // Test whether IP is at a conditional branch to self instruction.
  if ((ip[0] & 0xf0) != 0x70 || ip[1] != 0xfe)
    return false;

  // If so, test whether the condition is satisfied, in case we happened to
  // receive the signal at a not-taken branch to self.
  uint64_t eflags = uc->uc_mcontext.gregs[REG_EFL];
  switch (ip[0]) {
  case 0x70: // JO
    return eflags & X86_EFLAGS_OF;
  case 0x71: // JNO
    return !(eflags & X86_EFLAGS_OF);
  case 0x72: // JB
    return eflags & X86_EFLAGS_CF;
  case 0x73: // JAE
    return !(eflags & X86_EFLAGS_CF);
  case 0x74: // JE
    return eflags & X86_EFLAGS_ZF;
  case 0x75: // JNE
    return !(eflags & X86_EFLAGS_ZF);
  case 0x76: // JBE
    return (eflags & X86_EFLAGS_CF) || (eflags & X86_EFLAGS_ZF);
  case 0x77: // JA
    return !(eflags & X86_EFLAGS_CF) && !(eflags & X86_EFLAGS_ZF);
  case 0x78: // JS
    return eflags & X86_EFLAGS_SF;
  case 0x79: // JNS
    return !(eflags & X86_EFLAGS_SF);
  case 0x7A: // JP
    return eflags & X86_EFLAGS_PF;
  case 0x7B: // JNP
    return !(eflags & X86_EFLAGS_PF);
  case 0x7C: // JL
    return !!(eflags & X86_EFLAGS_SF) != !!(eflags & X86_EFLAGS_OF);
  case 0x7D: // JGE
    return !!(eflags & X86_EFLAGS_SF) == !!(eflags & X86_EFLAGS_OF);
  case 0x7E: // JLE
    return (eflags & X86_EFLAGS_ZF) ||
           !!(eflags & X86_EFLAGS_SF) != !!(eflags & X86_EFLAGS_OF);
  case 0x7F: // JG
    return !(eflags & X86_EFLAGS_ZF) &&
           !!(eflags & X86_EFLAGS_SF) == !!(eflags & X86_EFLAGS_OF);
  default:
    return false;
  }
}

static void SigprofHandler(int signo, siginfo_t *si, void *c) {
  if (__ubsan_is_trap_loop(c)) {
    __builtin_trap();
  }
}

void __ubsan_install_trap_loop_detection(void) {
  struct sigaction sa;
  sa.sa_sigaction = SigprofHandler;
  sigaction(SIGPROF, &sa, nullptr);

  struct itimerval timer;
  timer.it_value.tv_sec = 0;
  timer.it_value.tv_usec = 100000;
  timer.it_interval = timer.it_value;
  setitimer(ITIMER_PROF, &timer, NULL);
}

#else

int __ubsan_is_trap_loop(void *c) { return false; }
void __ubsan_install_trap_loop_detection(void) {}

#endif
