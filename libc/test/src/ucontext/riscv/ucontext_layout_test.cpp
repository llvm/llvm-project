//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Verifies that the layout of ucontext_t matches the signal frame set up by
/// the kernel. The test loads known values into as many registers as possible,
/// executes an ebreak instruction to raise a SIGTRAP, and checks that the
/// values show up in the right fields of the ucontext_t passed to the signal
/// handler.
///
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "hdr/types/siginfo_t.h"
#include "hdr/types/sigset_t.h"
#include "hdr/types/stack_t.h"
#include "hdr/types/struct_sigaction.h"
#include "include/llvm-libc-types/ucontext_t.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/macros/properties/cpu_features.h"
#include "src/signal/sigaction.h"
#include "src/signal/sigaddset.h"
#include "src/signal/sigaltstack.h"
#include "src/signal/sigemptyset.h"
#include "src/signal/sigismember.h"
#include "src/signal/sigprocmask.h"
#include "test/UnitTest/Test.h"

#if __riscv_xlen == 64
#define XLEN_LOAD "ld"
#define XLEN_STORE "sd"
#define XLEN_BYTES "8"
#elif __riscv_xlen == 32
#define XLEN_LOAD "lw"
#define XLEN_STORE "sw"
#define XLEN_BYTES "4"
#else
#error "Unsupported RISC-V architecture"
#endif

// Testing the floating point registers requires the D extension, as we use
// fld/fsd to load and save their 64-bit values. At present, the linux kernel
// does not support the Q extension, or the F extension alone.
#ifdef LIBC_TARGET_CPU_HAS_RISCV_FPU_DOUBLE
#define TEST_FP_REGS 1
#else
#define TEST_FP_REGS 0
#endif

// Load/store a register from/to element N of the array pointed to by t6.
#define LOAD_X(n) XLEN_LOAD " x" #n ", " #n "*" XLEN_BYTES "(t6)\n\t"
#define STORE_X(reg, n) XLEN_STORE " " #reg ", " #n "*" XLEN_BYTES "(t6)\n\t"
// Load fN from element N of the array pointed to by a1.
#define LOAD_F(n) "fld f" #n ", " #n "*8(a1)\n\t"

// Save/restore a register to/from an 8-byte stack slot.
#define SAVE(reg, slot) XLEN_STORE " " #reg ", " #slot "*8(sp)\n\t"
#define RESTORE(reg, slot) XLEN_LOAD " " #reg ", " #slot "*8(sp)\n\t"
#define SAVE_F(reg, slot) "fsd " #reg ", " #slot "*8(sp)\n\t"
#define RESTORE_F(reg, slot) "fld " #reg ", " #slot "*8(sp)\n\t"
// The stack frame holds 26 slots: ra, s0-s11, fs0-fs11 and fcsr. The size must
// be a multiple of 16 to keep the stack aligned.
#define FRAME_SIZE "26*8"

// clang-format mangles this badly
// clang-format off
// The callee-saved registers, and the stack slots we save them in.
#define CALLEE_SAVED_X(M)                                                      \
  M(ra, 0) M(s0, 1) M(s1, 2) M(s2, 3) M(s3, 4) M(s4, 5) M(s5, 6) M(s6, 7)      \
  M(s7, 8) M(s8, 9) M(s9, 10) M(s10, 11) M(s11, 12)
#define CALLEE_SAVED_F(M)                                                      \
  M(fs0, 13) M(fs1, 14) M(fs2, 15) M(fs3, 16) M(fs4, 17) M(fs5, 18)            \
  M(fs6, 19) M(fs7, 20) M(fs8, 21) M(fs9, 22) M(fs10, 23) M(fs11, 24)

// The register numbers.
#define ALL_REGS(M)                                                            \
  M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9) M(10) M(11) M(12) M(13)    \
  M(14) M(15) M(16) M(17) M(18) M(19) M(20) M(21) M(22) M(23) M(24) M(25)      \
  M(26) M(27) M(28) M(29) M(30) M(31)
// clang-format on

static int handler_calls;
static int received_signal;
static ucontext_t received_context;
static char alt_stack_buffer[SIGSTKSZ];

static void handler(int signal, siginfo_t *, void *context) {
  ucontext_t *uc = static_cast<ucontext_t *>(context);
  ++handler_calls;
  received_signal = signal;
  received_context = *uc;
  // Resume execution after the (four byte) ebreak instruction.
  uc->uc_mcontext.__gregs[REG_PC] += 4;
}

// Loads known values into the registers and executes an ebreak instruction:
// - xN is loaded from x[N]. The exceptions are sp, gp and tp, as the kernel and
//   the signal handler depend on them having sane values. Their current values
//   are stored into x[2], x[3] and x[4] instead. x[0] receives the address of
//   the ebreak instruction.
// - fN is loaded from f[N], and fcsr is set to `fcsr`.
// The callee-saved registers (and fcsr) are restored before returning.
[[gnu::naked]] static void
trap_with_known_registers(unsigned long * /*x*/,
                          const unsigned long long * /*f*/,
                          unsigned int /*fcsr*/) {
  // clang-format mangles this badly
  // clang-format off
  asm(
      // Save the callee-saved registers and fcsr.
      "addi sp, sp, -" FRAME_SIZE "\n\t"
      CALLEE_SAVED_X(SAVE)
#if TEST_FP_REGS
      CALLEE_SAVED_F(SAVE_F)
      "frcsr t0\n\t"
      SAVE(t0, 25)
#endif

      // Record the values of the registers we don't control. Use t6 (x31) to
      // hold the address of the array, as it is the last register we load.
      "mv t6, a0\n\t"
      "lla t0, 1f\n\t"
      STORE_X(t0, 0) STORE_X(sp, 2) STORE_X(gp, 3) STORE_X(tp, 4)

      // Load the known values. sp, gp and tp get the values we have just
      // stored, so they don't change, and the load into x0 has no effect.
#if TEST_FP_REGS
      ALL_REGS(LOAD_F)
      "fscsr a2\n\t"
#endif
      ALL_REGS(LOAD_X)

      // Raise the signal. Make sure the instruction does not get compressed,
      // so that the signal handler knows how to skip over it.
      ".option push\n\t"
      ".option norvc\n\t"
      "1: ebreak\n\t"
      ".option pop\n\t"

      // Restore the saved registers and return.
      CALLEE_SAVED_X(RESTORE)
#if TEST_FP_REGS
      CALLEE_SAVED_F(RESTORE_F)
      RESTORE(t0, 25)
      "fscsr t0\n\t"
#endif
      "addi sp, sp, " FRAME_SIZE "\n\t"
      "ret");
  // clang-format on
}

TEST(LlvmLibcUcontextLayoutTest, SignalContext) {
  // The values which the registers should hold when the signal is raised.
  // Element N of expected_x corresponds to register xN, except for element 0,
  // which corresponds to the pc. Every register gets a distinct and
  // recognizable value: each byte of xN is equal to N, and each byte of fN to
  // N + 0x80.
  unsigned long expected_x[32];
  unsigned long long expected_f[32];
  for (int i = 0; i < 32; ++i) {
    expected_x[i] = static_cast<unsigned long>(0x0101010101010101ULL * i);
    expected_f[i] = 0x0101010101010101ULL * (i + 0x80);
  }
  // Rounding mode RUP, and the NV, OF and NX exception flags.
  constexpr unsigned int EXPECTED_FCSR = 0x75;

  struct sigaction action = {};
  action.sa_sigaction = handler;
  // With SA_RESETHAND, the process crashes (instead of looping forever) if the
  // handler fails to skip over the ebreak instruction.
  action.sa_flags = SA_SIGINFO | SA_RESETHAND;
  ASSERT_EQ(LIBC_NAMESPACE::sigaction(SIGTRAP, &action, nullptr), 0);

  // Install an alternate signal stack to give uc_stack a recognizable value.
  // The handler does not run on it, as SA_ONSTACK is not set.
  stack_t alt_stack = {};
  alt_stack.ss_sp = alt_stack_buffer;
  alt_stack.ss_size = sizeof(alt_stack_buffer);
  stack_t old_stack;
  ASSERT_EQ(LIBC_NAMESPACE::sigaltstack(&alt_stack, &old_stack), 0);
  LIBC_NAMESPACE::cpp::scope_exit restore_stack(
      [&] { LIBC_NAMESPACE::sigaltstack(&old_stack, nullptr); });

  // Block some signals to give the signal mask a recognizable value.
  sigset_t blocked;
  ASSERT_EQ(LIBC_NAMESPACE::sigemptyset(&blocked), 0);
  ASSERT_EQ(LIBC_NAMESPACE::sigaddset(&blocked, SIGUSR1), 0);
  ASSERT_EQ(LIBC_NAMESPACE::sigaddset(&blocked, SIGUSR2), 0);
  ASSERT_EQ(LIBC_NAMESPACE::sigaddset(&blocked, SIGRTMAX), 0);
  sigset_t old_mask, expected_mask;
  ASSERT_EQ(LIBC_NAMESPACE::sigprocmask(SIG_BLOCK, &blocked, &old_mask), 0);
  LIBC_NAMESPACE::cpp::scope_exit restore_mask(
      [&] { LIBC_NAMESPACE::sigprocmask(SIG_SETMASK, &old_mask, nullptr); });
  ASSERT_EQ(LIBC_NAMESPACE::sigprocmask(SIG_BLOCK, nullptr, &expected_mask), 0);

  trap_with_known_registers(expected_x, expected_f, EXPECTED_FCSR);

  ASSERT_EQ(handler_calls, 1);
  ASSERT_EQ(received_signal, SIGTRAP);

  EXPECT_EQ(received_context.uc_flags, 0UL);
  EXPECT_EQ(received_context.uc_link, nullptr);
  EXPECT_EQ(received_context.uc_stack.ss_sp, alt_stack.ss_sp);
  EXPECT_EQ(received_context.uc_stack.ss_flags, alt_stack.ss_flags);
  EXPECT_EQ(received_context.uc_stack.ss_size, alt_stack.ss_size);
  for (int sig = 1; sig < NSIG; ++sig) {
    EXPECT_EQ(LIBC_NAMESPACE::sigismember(&received_context.uc_sigmask, sig),
              LIBC_NAMESPACE::sigismember(&expected_mask, sig))
        << "signal " << sig << "\n";
  }

  const mcontext_t &mcontext = received_context.uc_mcontext;
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(mcontext.__gregs[i], expected_x[i]) << "__gregs[" << i << "]\n";
  }
  EXPECT_EQ(mcontext.__gregs[REG_PC], expected_x[0]);
  EXPECT_EQ(mcontext.__gregs[REG_RA], expected_x[1]);
  EXPECT_EQ(mcontext.__gregs[REG_SP], expected_x[2]);
  EXPECT_EQ(mcontext.__gregs[REG_TP], expected_x[4]);
  EXPECT_EQ(mcontext.__gregs[REG_S0], expected_x[8]);
  EXPECT_EQ(mcontext.__gregs[REG_S1], expected_x[9]);
  EXPECT_EQ(mcontext.__gregs[REG_A0], expected_x[10]);
  EXPECT_EQ(mcontext.__gregs[REG_S2], expected_x[18]);

#if TEST_FP_REGS
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(mcontext.__fpregs.__d.__f[i], expected_f[i])
        << "__fpregs.__d.__f[" << i << "]\n";
  }
  EXPECT_EQ(mcontext.__fpregs.__d.__fcsr, EXPECTED_FCSR);
#endif
}
