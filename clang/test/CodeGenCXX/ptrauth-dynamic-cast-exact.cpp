// RUN: %clang_cc1 -I%S %s -triple arm64e-apple-darwin10 -O1 -fptrauth-calls -fptrauth-vtable-pointer-address-discrimination  -fptrauth-vtable-pointer-type-discrimination -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=CHECK

struct A {
  virtual ~A();
};
struct B {
  int foo;
  virtual ~B();
};
struct C final : A, B {
  virtual void f(){};
};
struct D final : B, A {
  virtual void f(){};
};

struct Offset {
  virtual ~Offset();
};
struct E {
  virtual ~E();
};
struct F final : Offset, E {
};
struct G {
  virtual ~G();
  int g;
};
struct H : E {
  int h;
};
struct I : E {
  int i;
};
struct J : virtual E {
  int j;
};
struct K : virtual E {
  int k;
};
struct L final : G, H, I, J, K {
  int l;
};
struct M final: G, private H { int m; };

// CHECK-LABEL: @_Z10exact_to_CP1A
C *exact_to_C(A *a) {
  // CHECK: [[UNAUTHED_VPTR:%.*]] = load ptr, ptr %a, align 8
  // CHECK: [[VPTR_ADDRI:%.*]] = ptrtoint ptr %a to i64
  // CHECK: [[VPTR_ADDR_DISC:%.*]] = tail call i64 @llvm.ptrauth.blend(i64 [[VPTR_ADDRI]], i64 62866)
  // CHECK: [[UNAUTHED_VPTRI:%.*]] = ptrtoint ptr [[UNAUTHED_VPTR]] to i64
  // CHECK: [[AUTHED_VPTRI:%.*]] = tail call i64 @llvm.ptrauth.auth(i64 [[UNAUTHED_VPTRI]], i32 2, i64 [[VPTR_ADDR_DISC]])
  // CHECK: [[IS_EXPECTED:%.*]] = icmp eq i64 [[AUTHED_VPTRI]], ptrtoint (ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTV1C, i64 16) to i64)
  // CHECK: br i1 [[IS_EXPECTED]], label %dynamic_cast.end, label %dynamic_cast.null
  // CHECK: [[NULL_CHECKED_RESULT:%.*]] = phi ptr [ %a, %dynamic_cast.notnull ], [ null, %dynamic_cast.null ]
  // CHECK: ret ptr [[NULL_CHECKED_RESULT]]
  return dynamic_cast<C*>(a);
}

// CHECK-LABEL: @_Z9exact_t_DP1A
D *exact_t_D(A *a) {
  // CHECK: dynamic_cast.notnull:
  // CHECK:   [[SRC_UNAUTHED_VPTR:%.*]] = load ptr, ptr %a
  // CHECK:   [[SRC_VPTR_ADDRI:%.*]] = ptrtoint ptr %a to i64
  // CHECK:   [[SRC_VPTR_DISC:%.*]] = tail call i64 @llvm.ptrauth.blend(i64 [[SRC_VPTR_ADDRI]], i64 62866)
  // CHECK:   [[SRC_UNAUTHED_VPTRI:%.*]] = ptrtoint ptr [[SRC_UNAUTHED_VPTR]] to i64
  // CHECK:   [[SRC_AUTHED_VPTRI:%.*]] = tail call i64 @llvm.ptrauth.auth(i64 [[SRC_UNAUTHED_VPTRI]], i32 2, i64 [[SRC_VPTR_DISC]])
  // CHECK:   [[SUCCESS:%.*]] = icmp eq i64 [[SRC_AUTHED_VPTRI]], ptrtoint (ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV1D, i64 56) to i64)
  // CHECK:   br i1 [[SUCCESS]], label %dynamic_cast.postauth.success, label %dynamic_cast.postauth.complete
  // CHECK: dynamic_cast.postauth.success:
  // CHECK:   [[ADJUSTED_THIS:%.*]] = getelementptr inbounds i8, ptr %a, i64 -16
  // CHECK:   [[ADJUSTED_UNAUTHED_VPTR:%.*]] = load ptr, ptr [[ADJUSTED_THIS]]
  // CHECK:   [[ADJUSTED_VPTR_ADDRI:%.*]] = ptrtoint ptr [[ADJUSTED_THIS]] to i64
  // CHECK:   [[ADJUSTED_VPTR_DISC:%.*]] = tail call i64 @llvm.ptrauth.blend(i64 [[ADJUSTED_VPTR_ADDRI]], i64 28965)
  // CHECK:   [[ADJUSTED_UNAUTHED_VPTRI:%.*]] = ptrtoint ptr [[ADJUSTED_UNAUTHED_VPTR]] to i64
  // CHECK:   [[ADJUSTED_AUTHED_VPTRI:%.*]] = tail call i64 @llvm.ptrauth.auth(i64 [[ADJUSTED_UNAUTHED_VPTRI]], i32 2, i64 [[ADJUSTED_VPTR_DISC]])
  // CHECK:   [[ADJUSTED_AUTHED_VPTR:%.*]] = inttoptr i64 [[ADJUSTED_AUTHED_VPTRI]] to ptr
  // CHECK:   br label %dynamic_cast.postauth.complete
  // CHECK: dynamic_cast.postauth.complete:
  // CHECK:   [[AUTHED_ADJUSTED_THIS:%.*]] = phi ptr [ [[ADJUSTED_THIS]], %dynamic_cast.postauth.success ], [ null, %dynamic_cast.notnull ]
  // CHECK:   br i1 [[SUCCESS]], label %dynamic_cast.end, label %dynamic_cast.null
  // CHECK: dynamic_cast.null:
  // CHECK:   br label %dynamic_cast.end
  // CHECK: dynamic_cast.end:
  // CHECK:   [[RESULT:%.*]] = phi ptr [ [[AUTHED_ADJUSTED_THIS]], %dynamic_cast.postauth.complete ], [ null, %dynamic_cast.null ]
  // CHECK:   ret ptr [[RESULT]]
  return dynamic_cast<D*>(a);
}

// CHECK-LABEL: @_Z11exact_multiP1E
L *exact_multi(E *e) {
  // CHECK: dynamic_cast.notnull:
  // CHECK:   [[VTABLE_ADDR:%.*]] = load ptr, ptr %e, align 8
  // CHECK:   [[THIS_ADDRI:%.*]] = ptrtoint ptr %e to i64
  // CHECK:   [[VTABLE_DISC:%.*]] = tail call i64 @llvm.ptrauth.blend(i64 [[THIS_ADDRI]], i64 12810)
  // CHECK:   [[VTABLE_ADDRI:%.*]] = ptrtoint ptr [[VTABLE_ADDR]] to i64
  // CHECK:   [[AUTHED_VTABLEI:%.*]] = tail call i64 @llvm.ptrauth.auth(i64 [[VTABLE_ADDRI]], i32 2, i64 [[VTABLE_DISC]])
  // CHECK:   [[AUTHED_VTABLE:%.*]] = inttoptr i64 [[AUTHED_VTABLEI]] to ptr
  // CHECK:   [[PRIMARY_BASE_OFFSET:%.*]] = getelementptr inbounds i8, ptr [[AUTHED_VTABLE]], i64 -16
  // CHECK:   %offset.to.top = load i64, ptr [[PRIMARY_BASE_OFFSET]]
  // CHECK:   [[ADJUSTED_THIS:%.*]] = getelementptr inbounds i8, ptr %e, i64 %offset.to.top
  // CHECK:   [[ADJUSTED_THIS_VTABLE:%.*]] = load ptr, ptr [[ADJUSTED_THIS]]
  // CHECK:   [[ADJUSTED_THIS_VTABLEI:%.*]] = ptrtoint ptr [[ADJUSTED_THIS_VTABLE]] to i64
  // CHECK:   [[ADJUSTED_THIS_STRIPPED_VTABLEI:%.*]] = tail call i64 @llvm.ptrauth.strip(i64 [[ADJUSTED_THIS_VTABLEI]], i32 0)
  // CHECK:   [[SUCCESS:%.*]] = icmp eq i64 [[ADJUSTED_THIS_STRIPPED_VTABLEI]], ptrtoint (ptr getelementptr inbounds nuw inrange(-24, 16) (i8, ptr @_ZTV1L, i64 24) to i64)
  // CHECK:   br i1 [[SUCCESS]], label %dynamic_cast.postauth.success, label %dynamic_cast.postauth.complete
  // CHECK: dynamic_cast.postauth.success:
  // CHECK:   [[ADJUSTED_THISI:%.*]] = ptrtoint ptr [[ADJUSTED_THIS]] to i64
  // CHECK:   [[DEST_DISC:%.*]] = tail call i64 @llvm.ptrauth.blend(i64 [[ADJUSTED_THISI]], i64 41434)
  // CHECK:   tail call i64 @llvm.ptrauth.auth(i64 [[ADJUSTED_THIS_VTABLEI]], i32 2, i64 [[DEST_DISC]])
  // CHECK:   br label %dynamic_cast.postauth.complete
  // CHECK: dynamic_cast.postauth.complete:
  // CHECK:   [[AUTHED_ADJUSTED_THIS:%.*]] = phi ptr [ [[ADJUSTED_THIS]], %dynamic_cast.postauth.success ], [ null, %dynamic_cast.notnull ]
  // CHECK:   br i1 [[SUCCESS]], label %dynamic_cast.end, label %dynamic_cast.null
  // CHECK: dynamic_cast.null:
  // CHECK:   br label %dynamic_cast.end
  // CHECK: dynamic_cast.end:
  // CHECK:   [[RESULT:%.*]] = phi ptr [ [[AUTHED_ADJUSTED_THIS]], %dynamic_cast.postauth.complete ], [ null, %dynamic_cast.null ]
  // CHECK:   ret ptr [[RESULT]]
  return dynamic_cast<L*>(e);
}

// CHECK-LABEL: @_Z19exact_invalid_multiP1H
M *exact_invalid_multi(H* d) {
  // CHECK: entry:
  // CHECK-NEXT:   ret ptr null
  return dynamic_cast<M*>(d);
}
