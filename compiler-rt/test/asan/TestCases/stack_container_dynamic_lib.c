// Test to demonstrate compile-time disabling of container-overflow checks
// in order to handle uninstrumented libraries
// UNSUPPORTED: target={{.*windows-.*}}

// Mimic a closed-source library compiled without ASan
// RUN: %clang_asan -fno-sanitize=address -DSHARED_LIB %s %fPIC -shared -o %t-so.so

// Mimic multiple files being linked into a single executable,
// %t-object.o and %t-main compiled seperately and then linked together
// RUN: %clang_asan -DMULTI_SOURCE %s -c -o %t-object.o
// RUN: %clang_asan %s -c -o %t-main.o
// RUN: %clang_asan -o %t %t-main.o %t-object.o %libdl
// RUN: not %run %t 2>&1 | FileCheck %s

// Disable container overflow checks at runtime using ASAN_OPTIONS=detect_container_overflow=0
// RUN: %env_asan_opts=detect_container_overflow=0 %run %t 2>&1 | FileCheck --check-prefix=CHECK-NO-CONTAINER-OVERFLOW %s

// RUN: %clang_asan -D__SANITIZER_DISABLE_CONTAINER_OVERFLOW__ -DMULTI_SOURCE %s -c -o %t-object.o
// RUN: %clang_asan -D__SANITIZER_DISABLE_CONTAINER_OVERFLOW__ %s -c -o %t-main.o
// RUN: %clang_asan -D__SANITIZER_DISABLE_CONTAINER_OVERFLOW__ -o %t %t-main.o %t-object.o %libdl
// RUN: %run %t 2>&1 | FileCheck --check-prefix=CHECK-NO-CONTAINER-OVERFLOW %s

#include <assert.h>
#include <sanitizer/common_interface_defs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Stack {
  int data[5];
  size_t size;
};

static void init(struct Stack *s) {
  s->size = 0;
#if __has_feature(address_sanitizer) &&                                        \
    !defined(__SANITIZER_DISABLE_CONTAINER_OVERFLOW__)
  // Mark entire storage as unaddressable initially
  __sanitizer_annotate_contiguous_container(s->data, s->data + 5, s->data + 5,
                                            s->data);
#endif
}

static void destroy(struct Stack *s) {
#if __has_feature(address_sanitizer) &&                                        \
    !defined(__SANITIZER_DISABLE_CONTAINER_OVERFLOW__)
  __sanitizer_annotate_contiguous_container(s->data, s->data + 5,
                                            s->data + s->size, s->data + 5);
#endif
}

static void push(struct Stack *s, int value) {
  assert(s->size < 5 && "Stack overflow");
#if __has_feature(address_sanitizer) &&                                        \
    !defined(__SANITIZER_DISABLE_CONTAINER_OVERFLOW__)
  __sanitizer_annotate_contiguous_container(
      s->data, s->data + 5, s->data + s->size, s->data + s->size + 1);
#endif
  s->data[s->size++] = value;
}

static int pop(struct Stack *s) {
  assert(s->size > 0 && "Cannot pop from empty stack");
  int result = s->data[--s->size];
#if __has_feature(address_sanitizer) &&                                        \
    !defined(__SANITIZER_DISABLE_CONTAINER_OVERFLOW__)
  __sanitizer_annotate_contiguous_container(
      s->data, s->data + 5, s->data + s->size + 1, s->data + s->size);
#endif
  return result;
}

#ifdef SHARED_LIB
// Mimics a closed-source library compiled without ASan

void push_value_to_stack(struct Stack *stack) { push(stack, 42); }
#else // SHARED_LIB

#  include <dlfcn.h>

typedef void (*push_func_t)(struct Stack *);

#  if defined(MULTI_SOURCE)
extern push_func_t push_value;

void do_push_value_to_stack(struct Stack *stack) {
  assert(push_value);
  push_value(stack);
}

#  else
push_func_t push_value = NULL;

void do_push_value_to_stack(struct Stack *stack);

int main(int argc, char *argv[]) {
  char path[4096];
  snprintf(path, sizeof(path), "%s-so.so", argv[0]);
  printf("Loading library: %s\n", path);

  void *lib = dlopen(path, RTLD_NOW);
  assert(lib);

  push_value = (push_func_t)dlsym(lib, "push_value_to_stack");
  assert(push_value);

  struct Stack stack;
  init(&stack);
  do_push_value_to_stack(&stack);

  // BOOM! uninstrumented library didn't update container bounds
  int value = pop(&stack);
  // CHECK: AddressSanitizer: container-overflow
  printf("Popped value: %d\n", value);
  assert(value == 42 && "Expected value 42");

  dlclose(lib);
  destroy(&stack);
  printf("SUCCESS\n");
  // CHECK-NO-CONTAINER-OVERFLOW: SUCCESS
  return 0;
}

#  endif // MULTI_SOURCE

#endif // SHARED_LIB
