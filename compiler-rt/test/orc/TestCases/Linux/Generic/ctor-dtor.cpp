// Check that constructors and destructors are run in the expected order.
//
// RUN: %clang -c -o %t.o %s
// RUN: %llvm_jitlink %t.o | FileCheck %s
//
// REQUIRES: system-linux && host-arch-compatible

// CHECK: <init_array.101>
// CHECK: <init_array.102>
// CHECK: <init_array.103>
// CHECK: <init_array>
// CHECK: <ctors.103>
// CHECK: <ctors.102>
// CHECK: <ctors.101>
// CHECK: <ctors>
// CHECK: <dtors>
// CHECK: <dtors.101>
// CHECK: <dtors.102>
// CHECK: <dtors.103>
// CHECK: <fini_array>
// CHECK: <fini_array.103>
// CHECK: <fini_array.102>
// CHECK: <fini_array.101>
#include <stdio.h>

typedef void (*ctor_t)(void);
typedef void (*dtor_t)(void);

__attribute__((constructor)) void init_array() { puts("<init_array>"); }

__attribute__((constructor(101))) void init_array_101() {
  puts("<init_array.101>");
}

__attribute__((constructor(102))) void init_array_102() {
  puts("<init_array.102>");
}
__attribute__((constructor(103))) void init_array_103() {
  puts("<init_array.103>");
}

static void ctors(void) { puts("<ctors>"); }
__attribute__((section(".ctors"), used)) static ctor_t ctors_ptr = ctors;

static void ctors_101(void) { puts("<ctors.101>"); }
__attribute__((section(".ctors.101"), used)) static ctor_t ctors_1_ptr =
    ctors_101;

static void ctors_102(void) { puts("<ctors.102>"); }
__attribute__((section(".ctors.102"), used)) static ctor_t ctors_2_ptr =
    ctors_102;

static void ctors_103(void) { puts("<ctors.103>"); }
__attribute__((section(".ctors.103"), used)) static ctor_t ctors_3_ptr =
    ctors_103;

__attribute__((destructor)) void fini_array() { puts("<fini_array>"); }

__attribute__((destructor(101))) void fini_array_101() {
  puts("<fini_array.101>");
}

__attribute__((destructor(102))) void fini_array_102() {
  puts("<fini_array.102>");
}

__attribute__((destructor(103))) void fini_array_103() {
  puts("<fini_array.103>");
}

static void dtors(void) { puts("<dtors>"); }
__attribute__((section(".dtors"), used)) static dtor_t dtors_ptr = dtors;

static void dtors_101(void) { puts("<dtors.101>"); }
__attribute__((section(".dtors.101"), used)) static dtor_t dtors_1_ptr =
    dtors_101;

static void dtors_102(void) { puts("<dtors.102>"); }
__attribute__((section(".dtors.102"), used)) static dtor_t dtors_2_ptr =
    dtors_102;

static void dtors_103(void) { puts("<dtors.103>"); }
__attribute__((section(".dtors.103"), used)) static dtor_t dtors_3_ptr =
    dtors_103;

int main(void) { return 0; }
