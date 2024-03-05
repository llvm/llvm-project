// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void ret_void() {}

//      CHECK: cir.func @_Z8ret_voidv()
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

int ret_non_void() {}

//      CHECK: cir.func @_Z12ret_non_voidv() -> !s32i
// CHECK-NEXT:   %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"]
// CHECK-NEXT:   cir.unreachable
// CHECK-NEXT: }
