// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int make_int();

int test() {
  const int &x = make_int();
  return x;
}

//      CHECK: cir.func @_Z4testv()
// CHECK-NEXT:   %{{.+}} = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:   %[[#TEMP_SLOT:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp0", init] {alignment = 4 : i64}
// CHECK-NEXT:   %[[#x:]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["x", init] {alignment = 8 : i64}
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     %[[#TEMP_VALUE:]] = cir.call @_Z8make_intv() : () -> !s32i
// CHECK-NEXT:     cir.store %[[#TEMP_VALUE]], %[[#TEMP_SLOT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.store %[[#TEMP_SLOT]], %[[#x]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
//      CHECK: }

int test_scoped() {
  int x = make_int();
  {
    const int &y = make_int();
    x = y;
  }
  return x;
}

//      CHECK: cir.func @_Z11test_scopedv()
// CHECK-NEXT:   %{{.+}} = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:   %{{.+}} = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
//      CHECK:   cir.scope {
// CHECK-NEXT:     %[[#TEMP_SLOT:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp0", init] {alignment = 4 : i64}
// CHECK-NEXT:     %[[#y:]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["y", init] {alignment = 8 : i64}
// CHECK-NEXT:     cir.scope {
// CHECK-NEXT:       %[[#TEMP_VALUE:]] = cir.call @_Z8make_intv() : () -> !s32i
// CHECK-NEXT:       cir.store %[[#TEMP_VALUE]], %[[#TEMP_SLOT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:     }
// CHECK-NEXT:     cir.store %[[#TEMP_SLOT]], %[[#y]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
//      CHECK:   }
//      CHECK: }
