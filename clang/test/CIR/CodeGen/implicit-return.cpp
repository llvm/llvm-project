// RUN: %clang_cc1 -O0 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CHECK-O0
// RUN: %clang_cc1 -O2 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CHECK-O2

void ret_void() {}

//      CHECK-O0: cir.func @_Z8ret_voidv()
// CHECK-O0-NEXT:   cir.return
// CHECK-O0-NEXT: }

//      CHECK-O2: cir.func @_Z8ret_voidv()
// CHECK-O2-NEXT:   cir.return
// CHECK-O2-NEXT: }

int ret_non_void() {}

//      CHECK-O0: cir.func @_Z12ret_non_voidv() -> !s32i
// CHECK-O0-NEXT:   %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"]
// CHECK-O0-NEXT:   cir.trap
// CHECK-O0-NEXT: }

//      CHECK-O2: cir.func @_Z12ret_non_voidv() -> !s32i
// CHECK-O2-NEXT:   %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"]
// CHECK-O2-NEXT:   cir.unreachable
// CHECK-O2-NEXT: }
