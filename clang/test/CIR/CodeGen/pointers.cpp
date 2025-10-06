// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Should generate basic pointer arithmetics.
void foo(int *iptr, char *cptr, unsigned ustride) {
  iptr + 2;
  // CHECK: %[[#STRIDE:]] = cir.const #cir.int<2> : !s32i
  // CHECK: cir.ptr_stride %{{.+}}, %[[#STRIDE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  cptr + 3;
  // CHECK: %[[#STRIDE:]] = cir.const #cir.int<3> : !s32i
  // CHECK: cir.ptr_stride %{{.+}}, %[[#STRIDE]] : (!cir.ptr<!s8i>, !s32i) -> !cir.ptr<!s8i>
  iptr - 2;
  // CHECK: %[[#STRIDE:]] = cir.const #cir.int<2> : !s32i
  // CHECK: %[[#NEGSTRIDE:]] = cir.unary(minus, %[[#STRIDE]]) : !s32i, !s32i
  // CHECK: cir.ptr_stride %{{.+}}, %[[#NEGSTRIDE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  cptr - 3;
  // CHECK: %[[#STRIDE:]] = cir.const #cir.int<3> : !s32i
  // CHECK: %[[#NEGSTRIDE:]] = cir.unary(minus, %[[#STRIDE]]) : !s32i, !s32i
  // CHECK: cir.ptr_stride %{{.+}}, %[[#NEGSTRIDE]] : (!cir.ptr<!s8i>, !s32i) -> !cir.ptr<!s8i>
  iptr + ustride;
  // CHECK: %[[#STRIDE:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
  // CHECK: cir.ptr_stride %{{.+}}, %[[#STRIDE]] : (!cir.ptr<!s32i>, !u32i) -> !cir.ptr<!s32i>

  // Must convert unsigned stride to a signed one.
  iptr - ustride;
  // CHECK: %[[#STRIDE:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
  // CHECK: %[[#SIGNSTRIDE:]] = cir.cast integral %[[#STRIDE]] : !u32i -> !s32i
  // CHECK: %[[#NEGSTRIDE:]] = cir.unary(minus, %[[#SIGNSTRIDE]]) : !s32i, !s32i
  // CHECK: cir.ptr_stride %{{.+}}, %[[#NEGSTRIDE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>

  4 + iptr;
  // CHECK: %[[#STRIDE:]] = cir.const #cir.int<4> : !s32i
  // CHECK: cir.ptr_stride %{{.+}}, %[[#STRIDE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>

  iptr++;
  // CHECK: %[[#STRIDE:]] = cir.const #cir.int<1> : !s32i
  // CHECK: cir.ptr_stride %{{.+}}, %[[#STRIDE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>

  iptr--;
  // CHECK: %[[#STRIDE:]] = cir.const #cir.int<-1> : !s32i
  // CHECK: cir.ptr_stride %{{.+}}, %[[#STRIDE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
}

void testPointerSubscriptAccess(int *ptr) {
// CHECK: testPointerSubscriptAccess
  ptr[1];
  // CHECK: %[[#STRIDE:]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[#PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CHECK: cir.ptr_stride %[[#PTR]], %[[#STRIDE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
}

void testPointerMultiDimSubscriptAccess(int **ptr) {
// CHECK: testPointerMultiDimSubscriptAccess
  ptr[1][2];
  // CHECK: %[[#STRIDE2:]] = cir.const #cir.int<2> : !s32i
  // CHECK: %[[#STRIDE1:]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[#PTR1:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: %[[#PTR2:]] = cir.ptr_stride %[[#PTR1]], %[[#STRIDE1]] : (!cir.ptr<!cir.ptr<!s32i>>, !s32i) -> !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: %[[#PTR3:]] = cir.load{{.*}} %[[#PTR2]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CHECK: cir.ptr_stride %[[#PTR3]], %[[#STRIDE2]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
}

// This test is meant to verify code that handles the 'p = nullptr + n' idiom
// used by some versions of glibc and gcc.  This is undefined behavior but
// it is intended there to act like a conversion from a pointer-sized integer
// to a pointer, and we would like to tolerate that.

#define NULLPTRINT ((int*)0)

// This should get the inttoptr instruction.
int *testGnuNullPtrArithmetic(unsigned n) {
// CHECK: testGnuNullPtrArithmetic
  return NULLPTRINT + n;
  // CHECK: %[[NULLPTR:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
  // CHECK: %[[N:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!u32i>, !u32i
  // CHECK: %[[RESULT:.*]] = cir.ptr_stride %[[NULLPTR]], %[[N]] : (!cir.ptr<!s32i>, !u32i) -> !cir.ptr<!s32i>
}
