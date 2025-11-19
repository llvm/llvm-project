// Test to demonstrate compile-time disabling of container-overflow checks
// in order to handle uninstrumented libraries
// UNSUPPORTED: target={{.*windows-msvc.*}}

// Mimic a closed-source library compiled without ASan
// RUN: %clangxx_asan -fno-sanitize=address -DSHARED_LIB %s %fPIC -shared -o %t-so.so

// Mimic multiple files being linked into a single executable,
// %t-object.o and %t-main compiled seperately and then linked together
// RUN: %clangxx_asan -DMULTI_SOURCE %s -c -o %t-object.o
// RUN: %clangxx_asan %s -c -o %t-main.o
// RUN: %clangxx_asan -o %t %t-main.o %t-object.o
// RUN: not %run %t 2>&1 | FileCheck %s

// Disable container overflow checks at runtime using ASAN_OPTIONS=detect_container_overflow=0
// RUN: %env_asan_opts=detect_container_overflow=0 %run %t 2>&1 | FileCheck --check-prefix=CHECK-NO-CONTAINER-OVERFLOW %s

// RUN: %clangxx_asan -D__SANITIZER_DISABLE_CONTAINER_OVERFLOW__ -DMULTI_SOURCE %s -c -o %t-object.o
// RUN: %clangxx_asan -D__SANITIZER_DISABLE_CONTAINER_OVERFLOW__ %s -c -o %t-main.o
// RUN: %clangxx_asan -D__SANITIZER_DISABLE_CONTAINER_OVERFLOW__ -o %t %t-main.o %t-object.o
// RUN: %run %t 2>&1 | FileCheck --check-prefix=CHECK-NO-CONTAINER-OVERFLOW %s

#include <assert.h>
#include <sanitizer/common_interface_defs.h>
#include <stdio.h>

template <typename T> class Stack {
private:
  T data[5];
  size_t size;

public:
  Stack() : size(0) {
#if __has_feature(address_sanitizer) && !__ASAN_DISABLE_CONTAINER_OVERFLOW__
    // Mark entire storage as unaddressable initially
    __sanitizer_annotate_contiguous_container(data, data + 5, data + 5, data);
#endif
  }

  ~Stack() {
#if __has_feature(address_sanitizer) && !__ASAN_DISABLE_CONTAINER_OVERFLOW__
    __sanitizer_annotate_contiguous_container(data, data + 5, data + size,
                                              data + 5);
#endif
  }

  void push(const T &value) {
    assert(size < 5 && "Stack overflow");
#if __has_feature(address_sanitizer) && !__ASAN_DISABLE_CONTAINER_OVERFLOW__
    __sanitizer_annotate_contiguous_container(data, data + 5, data + size,
                                              data + size + 1);
#endif
    data[size++] = value;
  }

  T pop() {
    assert(size > 0 && "Cannot pop from empty stack");
    T result = data[--size];
#if __has_feature(address_sanitizer) && !__ASAN_DISABLE_CONTAINER_OVERFLOW__
    __sanitizer_annotate_contiguous_container(data, data + 5, data + size + 1,
                                              data + size);
#endif
    return result;
  }
};

#ifdef SHARED_LIB
// Mimics a closed-source library compiled without ASan

extern "C" void push_value_to_stack(Stack<int> &stack) { stack.push(42); }
#else // SHARED_LIB

#  include <dlfcn.h>
#  include <string>

typedef void (*push_func_t)(Stack<int> &);

#  if defined(MULTI_SOURCE)
extern push_func_t push_value;

extern "C" void do_push_value_to_stack(Stack<int> &stack) {
  assert(push_value);
  push_value(stack);
}

#  else
push_func_t push_value = nullptr;

extern "C" void do_push_value_to_stack(Stack<int> &stack);

int main(int argc, char *argv[]) {
  std::string path = std::string(argv[0]) + "-so.so";
  printf("Loading library: %s\n", path.c_str());

  void *lib = dlopen(path.c_str(), RTLD_NOW);
  assert(lib);

  push_value = (push_func_t)dlsym(lib, "push_value_to_stack");
  assert(push_value);

  Stack<int> stack;
  do_push_value_to_stack(stack);

  // BOOM! uninstrumented library didn't update container bounds
  int value = stack.pop();
  // CHECK: AddressSanitizer: container-overflow
  printf("Popped value: %d\n", value);
  assert(value == 42 && "Expected value 42");

  dlclose(lib);
  printf("SUCCESS\n");
  // CHECK-NO-CONTAINER-OVERFLOW: SUCCESS
  return 0;
}

#  endif // MULTI_SOURCE

#endif // SHARED_LIB
