// RUN: %clang_cc1 -O0 -Wno-objc-root-class -fptrauth-intrinsics -fptrauth-calls -nostdsysteminc -triple arm64-apple-ios -emit-llvm -fptrauth-objc-interface-sel -o - %s | FileCheck --check-prefix=CHECK-AUTHENTICATED-SEL %s
// RUN: %clang_cc1 -O0 -Wno-objc-root-class -fptrauth-intrinsics -fptrauth-calls -nostdsysteminc -triple arm64-apple-ios -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-UNAUTHENTICATED-SEL %s

#include <ptrauth.h>
#define __ptrauth_objc_sel_override                 \
  __ptrauth(ptrauth_key_objc_sel_pointer, 1, 22467)

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
@public

  SEL __ptrauth_objc_sel_override _manual_sel_property;
}

@property SEL auto_sel_property;
@property const SEL const_auto_sel_property;
@property volatile SEL volatile_auto_sel_property;
@property SEL manual_sel_property;

@end

// CHECK-AUTHENTICATED-SEL-LABEL: define internal ptr @"\01-[Test test:]"
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 22466)
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 22466)
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.resign(i64 {{%.*}}, i32 3, i64 {{%.*}}, i32 3, i64 {{%.*}})
// CHECK-AUTHENTICATED-SEL: {{%.*}} = load volatile ptr, ptr {{%.*}}, align 8
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 22466)
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 22466)
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.resign(i64 {{%.*}}, i32 3, i64 {{%.*}}, i32 3, i64 {{%.*}})
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 22467)
// CHECK-AUTHENTICATED-SEL: {{%.*}} = ptrtoint ptr {{%.*}} to i64
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 22467)
// CHECK-AUTHENTICATED-SEL: {{%.*}} = ptrtoint ptr {{%.*}} to i64
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.resign(i64 {{%.*}}, i32 3, i64 {{%.*}}, i32 3, i64 {{%.*}})
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 22466)
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.auth(i64 {{%.*}}, i32 3, i64 {{%.*}})

// CHECK-AUTHENTICATED-SEL-LABEL: define internal ptr @"\01-[Test auto_sel_property]"
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 22466)
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.auth(i64 {{%.*}}, i32 3, i64 {{%.*}})

// CHECK-AUTHENTICATED-SEL-LABEL: define internal void @"\01-[Test setAuto_sel_property:]"
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 22466)
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.sign(i64 {{%.*}}, i32 3, i64 {{%.*}})

// CHECK-AUTHENTICATED-SEL-LABEL: define internal ptr @"\01-[Test const_auto_sel_property]"
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 22466)
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.auth(i64 {{%.*}}, i32 3, i64 {{%.*}})

// CHECK-AUTHENTICATED-SEL-LABEL: define internal void @"\01-[Test setConst_auto_sel_property:]"
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 22466)
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.sign(i64 {{%.*}}, i32 3, i64 {{%.*}})

// CHECK-AUTHENTICATED-SEL-LABEL: define internal ptr @"\01-[Test volatile_auto_sel_property]"
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 22466)
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.auth(i64 {{%.*}}, i32 3, i64 {{%.*}})

// CHECK-AUTHENTICATED-SEL-LABEL: define internal void @"\01-[Test setVolatile_auto_sel_property:]"
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 22466)
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.sign(i64 {{%.*}}, i32 3, i64 {{%.*}})

@implementation Test
- (SEL)test:(Test *)in {
  _auto_sel_property = in->_auto_sel_property;
  _volatile_auto_sel_property = in->_volatile_auto_sel_property;
  _manual_sel_property = in->_manual_sel_property;
  return _const_auto_sel_property;
}
@end

void auto_sel(Test *out, Test *in) {
  out->auto_sel = in->auto_sel;
}
// CHECK-AUTHENTICATED-SEL-LABEL: define void @auto_sel
// CHECK-AUTHENTICATED-SEL: [[DST_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_DST_ADDR:%.*]], i64 22466)
// CHECK-AUTHENTICATED-SEL: [[CAST_SRC_ADDR:%.*]] = ptrtoint ptr [[SRC_ADDR:%.*]] to i64
// CHECK-AUTHENTICATED-SEL: [[SRC_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_SRC_ADDR]], i64 22466)
// CHECK-AUTHENTICATED-SEL: [[SRC_SEL:%.*]] = ptrtoint ptr [[SRC_SEL_ADDR:%.*]] to i64
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.resign(i64 [[SRC_SEL]], i32 3, i64 [[DST_DESCRIMINATOR]], i32 3, i64 [[SRC_DESCRIMINATOR]])

// CHECK-UNAUTHENTICATED-SEL-LABEL: define void @auto_sel
SEL const_auto_sel(Test *in) {
  return in->const_auto_sel;
}

// CHECK-AUTHENTICATED-SEL-LABEL: define ptr @const_auto_sel
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 22466)
// CHECK-AUTHENTICATED-SEL: {{%.*}} = ptrtoint ptr {{%.*}} to i64
// CHECK-AUTHENTICATED-SEL: [[AUTHENTICATED:%.*]] = call i64 @llvm.ptrauth.auth(i64 {{%.*}}, i32 3, i64 {{%.*}})
// CHECK-AUTHENTICATED-SEL: [[RESULT:%.*]] = inttoptr i64 [[AUTHENTICATED]] to ptr

void volatile_auto_sel(Test *out, Test *in) {
  out->volatile_auto_sel = in->volatile_auto_sel;
}

// CHECK-AUTHENTICATED-SEL-LABEL: define void @volatile_auto_sel
// CHECK-AUTHENTICATED-SEL: [[DST_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_DST_ADDR:%.*]], i64 22466)
// CHECK-AUTHENTICATED-SEL: [[CAST_SRC_ADDR:%.*]] = ptrtoint ptr [[SRC_ADDR:%.*]] to i64
// CHECK-AUTHENTICATED-SEL: [[SRC_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_SRC_ADDR]], i64 22466)
// CHECK-AUTHENTICATED-SEL: [[SRC_SEL:%.*]] = ptrtoint ptr [[SRC_SEL_ADDR:%.*]] to i64
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.resign(i64 [[SRC_SEL]], i32 3, i64 [[DST_DESCRIMINATOR]], i32 3, i64 [[SRC_DESCRIMINATOR]])

void manual(Test *out, Test *in) {
  out->manual = in->manual;
}

// CHECK-AUTHENTICATED-SEL-LABEL: define void @manual
// CHECK-AUTHENTICATED-SEL: [[DST_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_DST_ADDR:%.*]], i64 22467)
// CHECK-AUTHENTICATED-SEL: [[CAST_SRC_ADDR:%.*]] = ptrtoint ptr [[SRC_ADDR:%.*]] to i64
// CHECK-AUTHENTICATED-SEL: [[SRC_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_SRC_ADDR]], i64 22467)
// CHECK-AUTHENTICATED-SEL: [[SRC_SEL:%.*]] = ptrtoint ptr [[SRC_SEL_ADDR:%.*]] to i64
// CHECK-AUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.resign(i64 [[SRC_SEL]], i32 3, i64 [[DST_DESCRIMINATOR]], i32 3, i64 [[SRC_DESCRIMINATOR]])

// CHECK-UNAUTHENTICATED-SEL-LABEL: define void @manual
// CHECK-UNAUTHENTICATED-SEL: [[DST_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_DST_ADDR:%.*]], i64 22467)
// CHECK-UNAUTHENTICATED-SEL: [[CAST_SRC_ADDR:%.*]] = ptrtoint ptr [[SRC_ADDR:%.*]] to i64
// CHECK-UNAUTHENTICATED-SEL: [[SRC_DESCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_SRC_ADDR]], i64 22467)
// CHECK-UNAUTHENTICATED-SEL: [[SRC_SEL:%.*]] = ptrtoint ptr [[SRC_SEL_ADDR:%.*]] to i64
// CHECK-UNAUTHENTICATED-SEL: {{%.*}} = call i64 @llvm.ptrauth.resign(i64 [[SRC_SEL]], i32 3, i64 [[DST_DESCRIMINATOR]], i32 3, i64 [[SRC_DESCRIMINATOR]])
