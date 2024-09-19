// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

#define NULL ((void *)0)

char *foo() {
  return (char*)NULL + 1;
}

// CHECK:  cir.func no_proto @foo()
// CHECK:    [[CONST_1:%[0-9]+]] = cir.const #cir.int<1> : !s32i
// CHECK:    {{.*}} = cir.cast(int_to_ptr, [[CONST_1]] : !s32i)
// CHECK:    cir.return
