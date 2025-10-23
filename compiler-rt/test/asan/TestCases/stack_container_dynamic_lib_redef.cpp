// Test to demonstrate compile-time disabling of container-overflow checks
// in order to handle uninstrumented libraries
//
// Explore three options discussed as implementable in the santizer headers to reduce the need
// for changes in library code if they include the sanitizer interface header sanitizer/common_interface_defs.h
// instead of having their own forward declaration
//
// - force inlined alternative body - minimizes the need for optimizations to reduce bloat
// - static declaration of alternate body - potential bloat if optimizer is not run
// - use of #define to remove calls completely

// Mimic a closed-source library compiled without ASan
// RUN: %clangxx_asan -fno-sanitize=address -DSHARED_LIB %s -dynamiclib -o %t-closedsource.dylib

// Mimic multiple files being linked into a single executable,
// %t-object.o and %t-main compiled seperately and then linked together
//
// -fsanitize=address with container overflow turned on (default)
// RUN: %clangxx_asan -DMULTI_SOURCE %s -c -o %t-object.o
// RUN: %clangxx_asan %s -c -o %t-main.o
// RUN: %clangxx_asan -o %t %t-main.o %t-object.o -framework %t-closedsource.dylib
// RUN: not %run %t 2>&1 | FileCheck %s
//
// -fsanitize=address with container overflow turned off using ALWAYS_INLINE
// RUN: %clangxx_asan -D__ASAN_DISABLE_CONTAINER_OVERFLOW__ -DMULTI_SOURCE %s -c -o %t-object.o
// RUN: %clangxx_asan -D__ASAN_DISABLE_CONTAINER_OVERFLOW__ %s -c -o %t-main.o
// RUN: %clangxx_asan -D__ASAN_DISABLE_CONTAINER_OVERFLOW__ -o %t %t-main.o %t-object.o -framework %t-closedsource.dylib
// RUN: %run %t 2>&1 | FileCheck --check-prefix=CHECK-NO-CONTAINER-OVERFLOW %s
//
// -fsanitize=address with container overflow turned off using static linkage
// RUN: %clangxx_asan -DUSE_STATIC -D__ASAN_DISABLE_CONTAINER_OVERFLOW__ -DMULTI_SOURCE %s -c -o %t-object.o
// RUN: %clangxx_asan -DUSE_STATIC -D__ASAN_DISABLE_CONTAINER_OVERFLOW__ %s -c -o %t-main.o
// RUN: %clangxx_asan -DUSE_STATIC -D__ASAN_DISABLE_CONTAINER_OVERFLOW__ -o %t %t-main.o %t-object.o -framework %t-closedsource.dylib
// RUN: %run %t 2>&1 | FileCheck --check-prefix=CHECK-NO-CONTAINER-OVERFLOW %s
//
// -fsanitize=address with container overflow turned off using #define to remove the function calls
// RUN: %clangxx_asan -DUSE_DEFINE -D__ASAN_DISABLE_CONTAINER_OVERFLOW__ -DMULTI_SOURCE %s -c -o %t-object.o
// RUN: %clangxx_asan -DUSE_DEFINE -D__ASAN_DISABLE_CONTAINER_OVERFLOW__ %s -c -o %t-main.o
// RUN: %clangxx_asan -DUSE_DEFINE -D__ASAN_DISABLE_CONTAINER_OVERFLOW__ -o %t %t-main.o %t-object.o -framework %t-closedsource.dylib
// RUN: %run %t 2>&1 | FileCheck --check-prefix=CHECK-NO-CONTAINER-OVERFLOW %s

#include <assert.h>
#include <stdio.h>

// Mimic sanitizer/common_interface_defs disabling the container overflow calls
#if !__has_feature(address_sanitizer) || __ASAN_DISABLE_CONTAINER_OVERFLOW__

#  if USE_DEFINE
#    define __sanitizer_annotate_contiguous_container(...)
#  else

// in this test match the extern "C" declaration as <string.h> pulls in copies of the
// declarations
extern "C" {
#    if USE_STATIC
static
#    else

// clone of the ALWAYS_INLINE macro
#      if defined(_MSC_VER)
#        define ALWAYS_INLINE __forceinline
#      else // _MSC_VER
#        define ALWAYS_INLINE inline __attribute__((always_inline))
#      endif // _MSC_VER

ALWAYS_INLINE
#    endif
    void __sanitizer_annotate_contiguous_container(const void *beg,
                                                   const void *end,
                                                   const void *old_mid,
                                                   const void *new_mid) {};
}
#  endif

#else

#  include <sanitizer/common_interface_defs.h>

#endif

// Mimic template based container library header
template <typename T> class Stack {
private:
  T data[5];
  size_t size;

public:
  Stack() : size(0) {
    // Mark entire storage as unaddressable initially
#if __has_feature(address_sanitizer)
    __sanitizer_annotate_contiguous_container(data, data + 5, data + 5, data);
#endif
  }

  ~Stack() {
#if __has_feature(address_sanitizer)
    __sanitizer_annotate_contiguous_container(data, data + 5, data + size,
                                              data + 5);
#endif
  }

  void push(const T &value) {
    assert(size < 5 && "Stack overflow");
#if __has_feature(address_sanitizer)
    __sanitizer_annotate_contiguous_container(data, data + 5, data + size,
                                              data + size + 1);
#endif
    data[size++] = value;
  }

  T pop() {
    assert(size > 0 && "Cannot pop from empty stack");
    T result = data[--size];
#if __has_feature(address_sanitizer)
    __sanitizer_annotate_contiguous_container(data, data + 5, data + size + 1,
                                              data + size);
#endif
    return result;
  }
};

#if defined(SHARED_LIB)
// Mimics a closed-source library compiled without ASan

extern "C" void push_value_to_stack(Stack<int> &stack) { stack.push(42); }

#elif defined(MULTI_SOURCE)

// Mimic multiple source files in a single project compiled seperately
extern "C" void push_value_to_stack(Stack<int> &stack);

extern "C" void do_push_value_to_stack(Stack<int> &stack) {
  push_value_to_stack(stack);
}

#else

extern "C" void do_push_value_to_stack(Stack<int> &stack);

#  include <string>

int main(int argc, char *argv[]) {

  Stack<int> stack;
  do_push_value_to_stack(stack);

  // BOOM! uninstrumented library didn't update container bounds
  int value = stack.pop();
  // CHECK: AddressSanitizer: container-overflow
  printf("Popped value: %d\n", value);
  assert(value == 42 && "Expected value 42");

  printf("SUCCESS\n");
  // CHECK-NO-CONTAINER-OVERFLOW: SUCCESS
  return 0;
}

#endif // SHARED_LIB
