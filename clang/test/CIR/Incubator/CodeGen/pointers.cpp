// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Should generate basic pointer arithmetics.
void foo(int *iptr, char *cptr, unsigned ustride) {
  *(iptr + 2) = 1;
  // CHECK: %[[#STRIDE:]] = cir.const #cir.int<2> : !s32i
  // CHECK: cir.ptr_stride inbounds %{{.+}}, %[[#STRIDE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  *(cptr + 3) = 1;
  // CHECK: %[[#STRIDE:]] = cir.const #cir.int<3> : !s32i
  // CHECK: cir.ptr_stride inbounds %{{.+}}, %[[#STRIDE]] : (!cir.ptr<!s8i>, !s32i) -> !cir.ptr<!s8i>
  *(iptr - 2) = 1;
  // CHECK: %[[#STRIDE:]] = cir.const #cir.int<2> : !s32i
  // CHECK: %[[#NEGSTRIDE:]] = cir.unary(minus, %[[#STRIDE]]) : !s32i, !s32i
  // CHECK: cir.ptr_stride inbounds %{{.+}}, %[[#NEGSTRIDE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  *(cptr - 3) = 1;
  // CHECK: %[[#STRIDE:]] = cir.const #cir.int<3> : !s32i
  // CHECK: %[[#NEGSTRIDE:]] = cir.unary(minus, %[[#STRIDE]]) : !s32i, !s32i
  // CHECK: cir.ptr_stride inbounds %{{.+}}, %[[#NEGSTRIDE]] : (!cir.ptr<!s8i>, !s32i) -> !cir.ptr<!s8i>
  *(iptr + ustride) = 1;
  // CHECK: %[[#STRIDE:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
  // CHECK: cir.ptr_stride inbounds|nuw %{{.+}}, %[[#STRIDE]] : (!cir.ptr<!s32i>, !u32i) -> !cir.ptr<!s32i>

  // Must convert unsigned stride to a signed one.
  *(iptr - ustride) = 1;
  // CHECK: %[[#STRIDE:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
  // CHECK: %[[#SIGNSTRIDE:]] = cir.cast integral %[[#STRIDE]] : !u32i -> !s32i
  // CHECK: %[[#NEGSTRIDE:]] = cir.unary(minus, %[[#SIGNSTRIDE]]) : !s32i, !s32i
  // CHECK: cir.ptr_stride inbounds %{{.+}}, %[[#NEGSTRIDE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
}

void testPointerSubscriptAccess(int *ptr) {
// CHECK: testPointerSubscriptAccess
  ptr[1] = 2;
  // CHECK: %[[#V1:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CHECK: %[[#V2:]] = cir.const #cir.int<1> : !s32i
  // CHECK: cir.ptr_stride %[[#V1]], %[[#V2]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
}

void testPointerMultiDimSubscriptAccess(int **ptr) {
// CHECK: testPointerMultiDimSubscriptAccess
  ptr[1][2] = 3;
  // CHECK: %[[#V1:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: %[[#V2:]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[#V3:]] = cir.ptr_stride %[[#V1]], %[[#V2]] : (!cir.ptr<!cir.ptr<!s32i>>, !s32i) -> !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: %[[#V4:]] = cir.load{{.*}} %[[#V3]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CHECK: %[[#V5:]] = cir.const #cir.int<2> : !s32i
  // CHECK: cir.ptr_stride %[[#V4]], %[[#V5]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
}
