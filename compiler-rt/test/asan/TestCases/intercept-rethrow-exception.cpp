// Regression test for
// https://bugs.llvm.org/show_bug.cgi?id=32434

// REQUIRES: shared_cxxabi

// RUN: %clangxx_asan -fexceptions -O0 %s -o %t
// RUN: %run %t

// The current implementation of this functionality requires special
// combination of libraries that are not used by default on NetBSD
// XFAIL: netbsd
// FIXME: Bug 42703
// XFAIL: solaris

#include <assert.h>
#include <exception>
#include <sanitizer/asan_interface.h>

namespace {

// Not instrumented because std::rethrow_exception is a [[noreturn]] function,
// for which the compiler would emit a call to __asan_handle_no_return which
// unpoisons the stack.
// We emulate here some code not compiled with asan. This function is not
// [[noreturn]] because the scenario we're emulating doesn't always throw. If it
// were [[noreturn]], the calling code would emit a call to
// __asan_handle_no_return.
void __attribute__((no_sanitize("address")))
uninstrumented_rethrow_exception(std::exception_ptr const &exc_ptr) {
  std::rethrow_exception(exc_ptr);
}

char *poisoned1;
char *poisoned2;

// Create redzones for stack variables in shadow memory and call
// std::rethrow_exception which should unpoison the entire stack.
void create_redzones_and_throw(std::exception_ptr const &exc_ptr) {
  char a[100];
  poisoned1 = a - 1;
  poisoned2 = a + sizeof(a);
  assert(__asan_address_is_poisoned(poisoned1));
  assert(__asan_address_is_poisoned(poisoned2));
  uninstrumented_rethrow_exception(exc_ptr);
}

} // namespace

// Check that std::rethrow_exception is intercepted by asan and the interception
// unpoisons the stack.
// If std::rethrow_exception is NOT intercepted, then calls to this function
// from instrumented code will still unpoison the stack because
// std::rethrow_exception is a [[noreturn]] function and any [[noreturn]]
// function call will be instrumented with __asan_handle_no_return.
// However, calls to std::rethrow_exception from UNinstrumented code will not
// unpoison the stack, so we need to intercept std::rethrow_exception to
// unpoison the stack.
int main() {
  // In some implementations of std::make_exception_ptr, e.g. libstdc++ prior to
  // gcc 7, this function calls __cxa_throw. The __cxa_throw is intercepted by
  // asan to unpoison the entire stack; since this test essentially tests that
  // the stack is unpoisoned by a call to std::rethrow_exception, we need to
  // generate the exception_ptr BEFORE we have the local variables poison the
  // stack.
  std::exception_ptr my_exception_ptr = std::make_exception_ptr("up");

  try {
    create_redzones_and_throw(my_exception_ptr);
  } catch(char const *) {
    assert(!__asan_region_is_poisoned(poisoned1, poisoned2 - poisoned1 + 1));
  }
}
