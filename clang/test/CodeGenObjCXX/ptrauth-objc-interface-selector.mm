// RUN: %clang_cc1 -O0 -Wobjc-root-class -fptrauth-intrinsics -fptrauth-calls -nostdsysteminc -triple arm64e-apple-ios -emit-llvm -fptrauth-objc-interface-sel -o - %s | FileCheck --check-prefix=CHECK-AUTHENTICATED-SEL %s
// RUN: %clang_cc1 -O0 -Wobjc-root-class -fptrauth-intrinsics -fptrauth-calls -nostdsysteminc -triple arm64e-apple-ios -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-UNAUTHENTICATED-SEL %s

#include <ptrauth.h>
#define __ptrauth_objc_sel_override                 \
  __ptrauth(ptrauth_key_objc_sel_pointer, 1, 22467)

extern "C" {

@interface Test {
@public
  SEL auto_sel;
@public
  const SEL const_auto_sel;
@public
  volatile SEL volatile_auto_sel;
@public
  SEL __ptrauth_objc_sel_override manual;
@public
  const SEL __ptrauth_objc_sel_override const_manual;
@public
  volatile SEL __ptrauth_objc_sel_override volatile_manual;
}

@end
#if __has_feature(ptrauth_objc_interface_sel)
typedef const SEL __ptrauth_objc_sel const_auto_sel_ptr_type;
const_auto_sel_ptr_type *const_auto_sel_ptr_type_test;
typedef volatile SEL __ptrauth_objc_sel volatile_auto_sel_ptr_type;
volatile_auto_sel_ptr_type *volatile_auto_sel_ptr_type_test;
#else
typedef const SEL const_auto_sel_ptr_type;
const_auto_sel_ptr_type *const_auto_sel_ptr_type_test;
typedef volatile SEL volatile_auto_sel_ptr_type;
volatile_auto_sel_ptr_type *volatile_auto_sel_ptr_type_test;
#endif

void auto_sel(Test *out, Test *in) {
  out->auto_sel = in->auto_sel;
}
// CHECK-AUTHENTICATED-SEL: define void @auto_sel
// CHECK-AUTHENTICATED-SEL: [[DST_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_DST_ADDR:%.*]], i64 22466)
// CHECK-AUTHENTICATED-SEL: [[CAST_SRC_ADDR:%.*]] = ptrtoint ptr [[SRC_ADDR:%.*]] to i64
// CHECK-AUTHENTICATED-SEL: [[SRC_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_SRC_ADDR]], i64 22466)
// CHECK-AUTHENTICATED-SEL: [[SRC_SEL:%.*]] = ptrtoint ptr [[SRC_SEL_ADDR:%.*]] to i64
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.resign(i64 [[SRC_SEL]], i32 3, i64 [[DST_DESCRIMINATOR]], i32 3, i64 [[SRC_DESCRIMINATOR]])

// CHECK-UNAUTHENTICATED-SEL: define void @auto_sel
SEL const_auto_sel(Test *in) {
  const_auto_sel_ptr_type_test = &in->const_auto_sel;
  return in->const_auto_sel;
}

// CHECK-AUTHENTICATED-SEL: define ptr @const_auto_sel
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 22466)
// CHECK-AUTHENTICATED-SEL: {{%.*}} = ptrtoint ptr {{%.*}} to i64
// CHECK-AUTHENTICATED-SEL: [[AUTHENTICATED:%.*]] = call i64 @llvm.ptrauth.auth(i64 {{%.*}}, i32 3, i64 {{%.*}})
// CHECK-AUTHENTICATED-SEL: [[RESULT:%.*]] = inttoptr i64 [[AUTHENTICATED]] to ptr

void volatile_auto_sel(Test *out, Test *in) {
  volatile_auto_sel_ptr_type_test = &in->volatile_auto_sel;
  out->volatile_auto_sel = in->volatile_auto_sel;
}

// CHECK-AUTHENTICATED-SEL: define void @volatile_auto_sel
// CHECK-AUTHENTICATED-SEL: [[DST_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_DST_ADDR:%.*]], i64 22466)
// CHECK-AUTHENTICATED-SEL: [[CAST_SRC_ADDR:%.*]] = ptrtoint ptr [[SRC_ADDR:%.*]] to i64
// CHECK-AUTHENTICATED-SEL: [[SRC_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_SRC_ADDR]], i64 22466)
// CHECK-AUTHENTICATED-SEL: [[SRC_SEL:%.*]] = ptrtoint ptr [[SRC_SEL_ADDR:%.*]] to i64
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.resign(i64 [[SRC_SEL]], i32 3, i64 [[DST_DESCRIMINATOR]], i32 3, i64 [[SRC_DESCRIMINATOR]])

void manual(Test *out, Test *in) {
  out->manual = in->manual;
}

// CHECK-AUTHENTICATED-SEL: define void @manual
// CHECK-AUTHENTICATED-SEL: [[DST_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_DST_ADDR:%.*]], i64 22467)
// CHECK-AUTHENTICATED-SEL: [[CAST_SRC_ADDR:%.*]] = ptrtoint ptr [[SRC_ADDR:%.*]] to i64
// CHECK-AUTHENTICATED-SEL: [[SRC_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_SRC_ADDR]], i64 22467)
// CHECK-AUTHENTICATED-SEL: [[SRC_SEL:%.*]] = ptrtoint ptr [[SRC_SEL_ADDR:%.*]] to i64
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.resign(i64 [[SRC_SEL]], i32 3, i64 [[DST_DESCRIMINATOR]], i32 3, i64 [[SRC_DESCRIMINATOR]])

// CHECK-UNAUTHENTICATED-SEL: define void @manual
// CHECK-UNAUTHENTICATED-SEL: [[DST_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_DST_ADDR:%.*]], i64 22467)
// CHECK-UNAUTHENTICATED-SEL: [[CAST_SRC_ADDR:%.*]] = ptrtoint ptr [[SRC_ADDR:%.*]] to i64
// CHECK-UNAUTHENTICATED-SEL: [[SRC_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_SRC_ADDR]], i64 22467)
// CHECK-UNAUTHENTICATED-SEL: [[SRC_SEL:%.*]] = ptrtoint ptr [[SRC_SEL_ADDR:%.*]] to i64
// CHECK-UNAUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.resign(i64 [[SRC_SEL]], i32 3, i64 [[DST_DESCRIMINATOR]], i32 3, i64 [[SRC_DESCRIMINATOR]])

}
