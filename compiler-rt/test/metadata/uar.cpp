// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=covered,uar && %t | FileCheck %s

// CHECK: metadata add version 1

// CHECK: empty: features=0 stack_args=0
void empty() {}

// CHECK: ellipsis: features=0 stack_args=0
void ellipsis(const char *fmt, ...) {
  volatile int x;
  x = 1;
}

// CHECK: non_empty_function: features=2 stack_args=0
void non_empty_function() {
  // Completely empty functions don't get uar metadata.
  volatile int x;
  x = 1;
}

// CHECK: no_stack_args: features=2 stack_args=0
void no_stack_args(long a0, long a1, long a2, long a3, long a4, long a5) {
  volatile int x;
  x = 1;
}

// CHECK: stack_args: features=2 stack_args=16
void stack_args(long a0, long a1, long a2, long a3, long a4, long a5, long a6) {
  volatile int x;
  x = 1;
}

// CHECK: more_stack_args: features=2 stack_args=32
void more_stack_args(long a0, long a1, long a2, long a3, long a4, long a5,
                     long a6, long a7, long a8) {
  volatile int x;
  x = 1;
}

// CHECK: struct_stack_args: features=2 stack_args=144
struct large {
  char x[131];
};
void struct_stack_args(large a) {
  volatile int x;
  x = 1;
}

#define FUNCTIONS                                                              \
  FN(empty);                                                                   \
  FN(ellipsis);                                                                \
  FN(non_empty_function);                                                      \
  FN(no_stack_args);                                                           \
  FN(stack_args);                                                              \
  FN(more_stack_args);                                                         \
  FN(struct_stack_args);                                                       \
  /**/

#include "common.h"
