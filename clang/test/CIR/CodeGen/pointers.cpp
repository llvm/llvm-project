// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Should generate basic pointer arithmetics.
void foo(int *iptr, char *cptr, unsigned ustride) {
  iptr + 2;
  // CHECK: %[[#STRIDE:]] = cir.const(#cir.int<2> : !s32i) : !s32i
  // CHECK: cir.ptr_stride(%{{.+}} : !cir.ptr<!s32i>, %[[#STRIDE]] : !s32i), !cir.ptr<!s32i>
  cptr + 3;
  // CHECK: %[[#STRIDE:]] = cir.const(#cir.int<3> : !s32i) : !s32i
  // CHECK: cir.ptr_stride(%{{.+}} : !cir.ptr<!s8i>, %[[#STRIDE]] : !s32i), !cir.ptr<!s8i>
  iptr - 2;
  // CHECK: %[[#STRIDE:]] = cir.const(#cir.int<2> : !s32i) : !s32i
  // CHECK: %[[#NEGSTRIDE:]] = cir.unary(minus, %[[#STRIDE]]) : !s32i, !s32i
  // CHECK: cir.ptr_stride(%{{.+}} : !cir.ptr<!s32i>, %[[#NEGSTRIDE]] : !s32i), !cir.ptr<!s32i>
  cptr - 3;
  // CHECK: %[[#STRIDE:]] = cir.const(#cir.int<3> : !s32i) : !s32i
  // CHECK: %[[#NEGSTRIDE:]] = cir.unary(minus, %[[#STRIDE]]) : !s32i, !s32i
  // CHECK: cir.ptr_stride(%{{.+}} : !cir.ptr<!s8i>, %[[#NEGSTRIDE]] : !s32i), !cir.ptr<!s8i>
  iptr + ustride;
  // CHECK: %[[#STRIDE:]] = cir.load %{{.+}} : cir.ptr <!u32i>, !u32i
  // CHECK: cir.ptr_stride(%{{.+}} : !cir.ptr<!s32i>, %[[#STRIDE]] : !u32i), !cir.ptr<!s32i>
  
  // Must convert unsigned stride to a signed one.
  iptr - ustride;
  // CHECK: %[[#STRIDE:]] = cir.load %{{.+}} : cir.ptr <!u32i>, !u32i
  // CHECK: %[[#SIGNSTRIDE:]] = cir.cast(integral, %[[#STRIDE]] : !u32i), !s32i
  // CHECK: %[[#NEGSTRIDE:]] = cir.unary(minus, %[[#SIGNSTRIDE]]) : !s32i, !s32i
  // CHECK: cir.ptr_stride(%{{.+}} : !cir.ptr<!s32i>, %[[#NEGSTRIDE]] : !s32i), !cir.ptr<!s32i>
}

void testPointerSubscriptAccess(int *ptr) {
// CHECK: testPointerSubscriptAccess
  ptr[1];
  // CHECK: %[[#V1:]] = cir.load %{{.+}} : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CHECK: %[[#V2:]] = cir.const(#cir.int<1> : !s32i) : !s32i
  // CHECK: cir.ptr_stride(%[[#V1]] : !cir.ptr<!s32i>, %[[#V2]] : !s32i), !cir.ptr<!s32i>
}

void testPointerMultiDimSubscriptAccess(int **ptr) {
// CHECK: testPointerMultiDimSubscriptAccess
  ptr[1][2];
  // CHECK: %[[#V1:]] = cir.load %{{.+}} : cir.ptr <!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: %[[#V2:]] = cir.const(#cir.int<1> : !s32i) : !s32i
  // CHECK: %[[#V3:]] = cir.ptr_stride(%[[#V1]] : !cir.ptr<!cir.ptr<!s32i>>, %[[#V2]] : !s32i), !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: %[[#V4:]] = cir.load %[[#V3]] : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CHECK: %[[#V5:]] = cir.const(#cir.int<2> : !s32i) : !s32i
  // CHECK: cir.ptr_stride(%[[#V4]] : !cir.ptr<!s32i>, %[[#V5]] : !s32i), !cir.ptr<!s32i>
}
