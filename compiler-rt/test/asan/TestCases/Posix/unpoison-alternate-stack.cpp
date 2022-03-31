// Tests that __asan_handle_no_return properly unpoisons the signal alternate
// stack.

// Don't optimize, otherwise the variables which create redzones might be
// dropped.
// RUN: %clangxx_asan -fexceptions -O0 %s -o %t -pthread
// RUN: %run %t

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include <sanitizer/asan_interface.h>

namespace {

struct TestContext {
  char *LeftRedzone;
  char *RightRedzone;
  std::jmp_buf JmpBuf;
};

TestContext defaultStack;
TestContext signalStack;

// Create a new stack frame to ensure that logically, the stack frame should be
// unpoisoned when the function exits. Exit is performed via jump, not return,
// such that we trigger __asan_handle_no_return and not ordinary unpoisoning.
template <class Jump>
void __attribute__((noinline)) poisonStackAndJump(TestContext &c, Jump jump) {
  char Blob[100]; // This variable must not be optimized out, because we use it
                  // to create redzones.

  c.LeftRedzone = Blob - 1;
  c.RightRedzone = Blob + sizeof(Blob);

  assert(__asan_address_is_poisoned(c.LeftRedzone));
  assert(__asan_address_is_poisoned(c.RightRedzone));

  // Jump to avoid normal cleanup of redzone markers. Instead,
  // __asan_handle_no_return is called which unpoisons the stacks.
  jump();
}

void testOnCurrentStack() {
  TestContext c;

  if (0 == setjmp(c.JmpBuf))
    poisonStackAndJump(c, [&] { longjmp(c.JmpBuf, 1); });

  assert(0 == __asan_region_is_poisoned(c.LeftRedzone,
                                        c.RightRedzone - c.LeftRedzone));
}

bool isOnSignalStack() {
  stack_t Stack;
  sigaltstack(nullptr, &Stack);
  return Stack.ss_flags == SS_ONSTACK;
}

void signalHandler(int, siginfo_t *, void *) {
  assert(isOnSignalStack());

  // test on signal alternate stack
  testOnCurrentStack();

  // test unpoisoning when jumping between stacks
  poisonStackAndJump(signalStack, [] { longjmp(defaultStack.JmpBuf, 1); });
}

void setSignalAlternateStack(void *AltStack) {
  sigaltstack((stack_t const *)AltStack, nullptr);

  struct sigaction Action = {};
  Action.sa_sigaction = signalHandler;
  Action.sa_flags = SA_SIGINFO | SA_NODEFER | SA_ONSTACK;
  sigemptyset(&Action.sa_mask);

  sigaction(SIGUSR1, &Action, nullptr);
}

// Main test function.
// Must be run on another thread to be able to control memory placement between
// default stack and alternate signal stack.
// If the alternate signal stack is placed in close proximity before the
// default stack, __asan_handle_no_return might unpoison both, even without
// being aware of the signal alternate stack.
// We want to test reliably that __asan_handle_no_return can properly unpoison
// the signal alternate stack.
void *threadFun(void *AltStack) {
  // first test on default stack (sanity check), no signal alternate stack set
  testOnCurrentStack();

  setSignalAlternateStack(AltStack);

  // test on default stack again, but now the signal alternate stack is set
  testOnCurrentStack();

  // set up jump to test unpoisoning when jumping between stacks
  if (0 == setjmp(defaultStack.JmpBuf))
    // Test on signal alternate stack, via signalHandler
    poisonStackAndJump(defaultStack, [] { raise(SIGUSR1); });

  assert(!isOnSignalStack());

  assert(0 == __asan_region_is_poisoned(
                  defaultStack.LeftRedzone,
                  defaultStack.RightRedzone - defaultStack.LeftRedzone));

  assert(0 == __asan_region_is_poisoned(
                  signalStack.LeftRedzone,
                  signalStack.RightRedzone - signalStack.LeftRedzone));

  return nullptr;
}

} // namespace

// Check that __asan_handle_no_return properly unpoisons a signal alternate
// stack.
// __asan_handle_no_return tries to determine the stack boundaries and
// unpoisons all memory inside those. If this is not done properly, redzones for
// variables on can remain in shadow memory which might lead to false positive
// reports when the stack is reused.
int main() {
  size_t const PageSize = sysconf(_SC_PAGESIZE);
  // The Solaris defaults of 4k (32-bit) and 8k (64-bit) are too small.
  size_t const MinStackSize = std::max<size_t>(PTHREAD_STACK_MIN, 16 * 1024);
  // To align the alternate stack, we round this up to page_size.
  size_t const DefaultStackSize =
      (MinStackSize - 1 + PageSize) & ~(PageSize - 1);
  // The alternate stack needs a certain size, or the signal handler segfaults.
  size_t const AltStackSize = 10 * PageSize;
  size_t const MappingSize = DefaultStackSize + AltStackSize;
  // Using mmap guarantees proper alignment.
  void *const Mapping = mmap(nullptr, MappingSize,
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS,
                             -1, 0);

  stack_t AltStack = {};
  AltStack.ss_sp = (char *)Mapping + DefaultStackSize;
  AltStack.ss_flags = 0;
  AltStack.ss_size = AltStackSize;

  pthread_t Thread;
  pthread_attr_t ThreadAttr;
  pthread_attr_init(&ThreadAttr);
  pthread_attr_setstack(&ThreadAttr, Mapping, DefaultStackSize);
  pthread_create(&Thread, &ThreadAttr, &threadFun, (void *)&AltStack);

  pthread_join(Thread, nullptr);

  munmap(Mapping, MappingSize);
}
