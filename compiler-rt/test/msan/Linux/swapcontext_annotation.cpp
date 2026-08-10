// RUN: %clangxx_msan -O0 %s -o %t && %run %t

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <ucontext.h>
#include <unistd.h>

#include <sanitizer/msan_interface.h>

namespace {

const int kStackSize = 1 << 20;
char fiber_stack[kStackSize] = {};

ucontext_t main_ctx;
ucontext_t fiber_ctx;

void fiber() {
  printf("%s: entering fiber\n", __FUNCTION__);

  // This fiber was switched into from main. Verify the details of main's stack
  // have been populated by MSAN.
  const void *previous_stack_bottom = nullptr;
  size_t previous_stack_size = 0;
  __msan_finish_switch_fiber(&previous_stack_bottom, &previous_stack_size);
  assert(previous_stack_bottom != nullptr);
  assert(previous_stack_size != 0);

  printf("%s: implicitly swapcontext to main\n", __FUNCTION__);
  __msan_start_switch_fiber(previous_stack_bottom, previous_stack_size);
}

} // namespace

// Set up a fiber, switch to it, and switch back, invoking __msan_*_switch_fiber
// functions along the way. At each step, validate the correct stack addresses and
// sizes are returned from those functions.
int main(int argc, char **argv) {
  if (getcontext(&fiber_ctx) == -1) {
    perror("getcontext");
    _exit(1);
  }
  fiber_ctx.uc_stack.ss_sp = fiber_stack;
  fiber_ctx.uc_stack.ss_size = sizeof(fiber_stack);
  fiber_ctx.uc_link = &main_ctx;
  makecontext(&fiber_ctx, fiber, 0);

  // Tell MSAN a fiber switch is about to occur, then perform the switch
  printf("%s: swapcontext to fiber\n", __FUNCTION__);
  __msan_start_switch_fiber(fiber_stack, kStackSize);
  if (swapcontext(&main_ctx, &fiber_ctx) == -1) {
    perror("swapcontext");
    _exit(1);
  }

  // The fiber switched to above now switched back here. Tell MSAN that switch
  // is complete and verify the fiber details return by MSAN are correct.
  const void *previous_stack_bottom = nullptr;
  size_t previous_stack_size = 0;
  __msan_finish_switch_fiber(&previous_stack_bottom, &previous_stack_size);
  assert(previous_stack_bottom == fiber_stack);
  assert(previous_stack_size == kStackSize);

  printf("%s: exiting\n", __FUNCTION__);

  return 0;
}
