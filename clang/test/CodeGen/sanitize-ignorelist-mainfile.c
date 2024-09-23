/// Test mainfile in a sanitizer special case list.
// RUN: rm -rf %t && split-file %s %t
// RUN: %clang_cc1 -emit-llvm -triple x86_64 -fsanitize=address,alignment %t/a.c -o - | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: %clang_cc1 -emit-llvm -triple x86_64 -fsanitize=address,alignment -fsanitize-ignorelist=%t/a.list %t/a.c -o - | FileCheck %s --check-prefixes=CHECK,IGNORE
// RUN: %clang_cc1 -emit-llvm -triple x86_64 -fsanitize=address,alignment -fsanitize-ignorelist=%t/b.list %t/a.c -o - | FileCheck %s --check-prefixes=CHECK,IGNORE

//--- a.list
mainfile:*a.c

//--- b.list
[address]
mainfile:*a.c

[alignment]
mainfile:*.c

//--- a.h
int global_h;

static inline int load(int *x) {
  return *x;
}

//--- a.c
#include "a.h"

int global_c;

int foo(void *x) {
  return load(x);
}

// DEFAULT:     @___asan_gen_{{.*}} = {{.*}} c"global_h\00"
// DEFAULT:     @___asan_gen_{{.*}} = {{.*}} c"global_c\00"
// IGNORE-NOT:  @___asan_gen_

// CHECK-LABEL: define {{.*}}@load(
// DEFAULT:       call void @__ubsan_handle_type_mismatch_v1_abort(
// DEFAULT:       call void @__asan_report_load4(
// IGNORE-NOT:    call void @__ubsan_handle_type_mismatch_v1_abort(
// IGNORE-NOT:    call void @__asan_report_load4(
