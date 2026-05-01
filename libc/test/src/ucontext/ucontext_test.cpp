//===-- Unittests for ucontext routines -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ucontext/getcontext.h"
#include "src/ucontext/makecontext.h"
#include "src/ucontext/setcontext.h"
#include "src/ucontext/swapcontext.h"

#include "src/signal/sigaddset.h"
#include "src/signal/sigemptyset.h"
#include "src/signal/sigprocmask.h"

#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/signal-macros.h"

namespace LIBC_NAMESPACE {

static bool is_signal_set(const sigset_t *set, int signum) {
  // NSIG is 64, sigset_t is an array of unsigned long.
  // Signum is 1-indexed.
  int word = (signum - 1) / (sizeof(unsigned long) * 8);
  int bit = (signum - 1) % (sizeof(unsigned long) * 8);
  return (set->__signals[word] & (1UL << bit)) != 0;
}

volatile int jumped = 0;
TEST(LlvmLibcUcontextTest, BasicStubTest) {
  ucontext_t ctx;
  ASSERT_EQ(getcontext(&ctx), 0);
  if (!jumped) {
    jumped = 1;
    setcontext(&ctx);
    ASSERT_TRUE(false && "setcontext should not return on success");
  }
}

ucontext_t old_ctx, new_ctx;
volatile int swap_called = 0;

void swap_func() {
  swap_called = 1;
  setcontext(&old_ctx);
}

TEST(LlvmLibcUcontextTest, SwapcontextTest) {
  getcontext(&new_ctx);
  constexpr size_t STACK_SIZE = 8192;
  char stack[STACK_SIZE];
  new_ctx.uc_stack.ss_sp = stack;
  new_ctx.uc_stack.ss_size = sizeof(stack);
  makecontext(&new_ctx, swap_func, 0);

  swapcontext(&old_ctx, &new_ctx);

  ASSERT_EQ(swap_called, 1);
}

ucontext_t old_ctx_args, new_ctx_args;
volatile int makecontext_args_called = 0;
volatile int arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8;

void args_func(int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8) {
  makecontext_args_called = 1;
  arg1 = a1;
  arg2 = a2;
  arg3 = a3;
  arg4 = a4;
  arg5 = a5;
  arg6 = a6;
  arg7 = a7;
  arg8 = a8;
  setcontext(&old_ctx_args);
}

TEST(LlvmLibcUcontextTest, MakecontextArgsTest) {
  getcontext(&new_ctx_args);
  constexpr size_t STACK_SIZE = 8192;
  char stack[STACK_SIZE];
  new_ctx_args.uc_stack.ss_sp = stack;
  new_ctx_args.uc_stack.ss_size = sizeof(stack);

  // Cast function pointer to void(*)(void) as required by makecontext
  using func_t = void (*)(void);
  auto func = reinterpret_cast<func_t>(args_func);

  makecontext(&new_ctx_args, func, 8, 11, 22, 33, 44, 55, 66, 77, 88);

  swapcontext(&old_ctx_args, &new_ctx_args);

  ASSERT_EQ(makecontext_args_called, 1);
  ASSERT_EQ(arg1, 11);
  ASSERT_EQ(arg2, 22);
  ASSERT_EQ(arg3, 33);
  ASSERT_EQ(arg4, 44);
  ASSERT_EQ(arg5, 55);
  ASSERT_EQ(arg6, 66);
  ASSERT_EQ(arg8, 88);
}

ucontext_t old_ctx_return, new_ctx_return;
volatile int makecontext_return_called = 0;

void return_func() { makecontext_return_called = 1; }

TEST(LlvmLibcUcontextTest, MakecontextReturnTest) {
  getcontext(&new_ctx_return);
  constexpr size_t STACK_SIZE = 8192;
  char stack[STACK_SIZE];
  new_ctx_return.uc_stack.ss_sp = stack;
  new_ctx_return.uc_stack.ss_size = sizeof(stack);
  new_ctx_return.uc_link = &old_ctx_return;

  using func_t = void (*)(void);
  auto func = reinterpret_cast<func_t>(return_func);

  makecontext(&new_ctx_return, func, 0);

  swapcontext(&old_ctx_return, &new_ctx_return);

  ASSERT_EQ(makecontext_return_called, 1);
}

TEST(LlvmLibcUcontextTest, SignalMaskTest) {
  sigset_t set, old_set;
  sigemptyset(&set);
  sigaddset(&set, SIGUSR1);

  // Set mask to [SIGUSR1] using sigprocmask
  sigprocmask(SIG_SETMASK, &set, &old_set);

  ucontext_t ctx;
  getcontext(&ctx);

  // Verify that getcontext captured the mask
  ASSERT_TRUE(is_signal_set(&ctx.uc_sigmask, SIGUSR1));
  ASSERT_FALSE(is_signal_set(&ctx.uc_sigmask, SIGUSR2));

  sigset_t new_set;
  static volatile int mask_jumped = 0;
  if (mask_jumped == 0) {
    mask_jumped = 1;
    sigemptyset(&new_set);
    sigaddset(&new_set, SIGUSR2);
    sigprocmask(SIG_SETMASK, &new_set, nullptr);

    setcontext(&ctx);
  }

  // Check current mask
  sigset_t current;
  sigprocmask(SIG_BLOCK, nullptr, &current);

  // Restore original mask for clean state
  sigprocmask(SIG_SETMASK, &old_set, nullptr);

  ASSERT_TRUE(is_signal_set(&current, SIGUSR1));
  ASSERT_FALSE(is_signal_set(&current, SIGUSR2));
}

} // namespace LIBC_NAMESPACE
