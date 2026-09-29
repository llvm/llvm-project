// RUN: %clang_cc1 -Wno-everything -fblocks -fptrauth-intrinsics -triple arm64-apple-ios -fobjc-runtime=ios-12.2 -emit-llvm -no-enable-noundef-analysis -fobjc-arc -O2 -disable-llvm-passes -o - %s | FileCheck %s

#if __has_feature(ptrauth_objc_signable_class)
struct TestStruct {
  __ptrauth(2, 1, 1234) Class isa;
};

@interface TestClass {
@public
  __ptrauth(2, 1, 1234) Class isa;
}
@end

struct TestConstStruct {
  __ptrauth(2, 1, 1234) const Class isa;
  __ptrauth(2, 1, 1234) volatile Class visa;
};

@interface TestConstClass {
@public
  __ptrauth(2, 1, 1234) const Class isa;
  __ptrauth(2, 1, 1234) volatile Class visa;
}
@end

// CHECK-LABEL: define void @setTestStructIsa(ptr %t, ptr %c) #0 {
void setTestStructIsa(struct TestStruct *t, Class c) {
  t->isa = c;
  // CHECK: [[T_ADDR:%.*]] = alloca ptr, align 8
  // CHECK: [[C_ADDR:%.*]] = alloca ptr, align 8
  // CHECK: store ptr %c, ptr [[C_ADDR]], align 8
  // CHECK: [[ISA_SLOT:%.*]] = getelementptr inbounds nuw %struct.TestStruct, ptr %0, i32 0, i32 0
  // CHECK: [[C:%.*]] = load ptr, ptr %c.addr, align 8
  // CHECK: [[CAST_ISA_SLOT:%.*]] = ptrtoint ptr [[ISA_SLOT]] to i64
  // CHECK: [[BLENDED_VALUE:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_ISA_SLOT]], i64 1234)
  // CHECK: [[CAST_C:%.*]] = ptrtoint ptr [[C]] to i64
  // CHECK: [[AUTHENTICATED:%.*]] = call i64 @llvm.ptrauth.sign(i64 [[CAST_C]], i32 2, i64 [[BLENDED_VALUE]])
}

// CHECK-LABEL: define void @setTestClassIsa(ptr %t, ptr %c) #0 {
void setTestClassIsa(TestClass *t, Class c) {
  t->isa = c;
  // CHECK: [[T_ADDR:%.*]] = alloca ptr, align 8
  // CHECK: [[C_ADDR:%.*]] = alloca ptr, align 8
  // CHECK: store ptr %c, ptr [[C_ADDR]], align 8
  // CHECK: [[T:%.*]] = load ptr, ptr [[T_ADDR]], align 8
  // CHECK: [[IVAR_OFFSET32:%.*]] = load i32, ptr @"OBJC_IVAR_$_TestClass.isa", align 8
  // CHECK: [[IVAR_OFFSET64:%.*]] = sext i32 [[IVAR_OFFSET32]] to i64
  // CHECK: [[ADDED_PTR:%.*]] = getelementptr inbounds i8, ptr %1, i64 [[IVAR_OFFSET64]]
  // CHECK: [[C_VALUE:%.*]] = load ptr, ptr [[C_ADDR]], align 8
  // CHECK: [[CAST_ISA_SLOT:%.*]] = ptrtoint ptr [[ADDED_PTR]] to i64
  // CHECK: [[BLENDED_VALUE:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_ISA_SLOT]], i64 1234)
  // CHECK: [[CAST_C_VALUE:%.*]] = ptrtoint ptr [[C_VALUE]] to i64
  // CHECK: [[SIGNED:%.*]] = call i64 @llvm.ptrauth.sign(i64 [[CAST_C_VALUE]], i32 2, i64 [[BLENDED_VALUE]])
}

// CHECK-LABEL: define ptr @getTestStructIsa(ptr %t) #0 {
Class getTestStructIsa(struct TestStruct *t) {
  return t->isa;
  // CHECK: [[T_ADDR:%.*]] = alloca ptr, align 8
  // CHECK: [[T_VALUE:%.*]] = load ptr, ptr [[T_ADDR]], align 8
  // CHECK: [[ISA_SLOT:%.*]] = getelementptr inbounds nuw %struct.TestStruct, ptr [[T_VALUE]], i32 0, i32 0
  // CHECK: [[ISA_VALUE:%.*]] = load ptr, ptr [[ISA_SLOT]], align 8
  // CHECK: [[CAST_ISA_SLOT:%.*]] = ptrtoint ptr %isa to i64
  // CHECK: [[BLENDED_VALUE:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_ISA_SLOT]], i64 1234)
  // CHECK: [[CAST_ISA_VALUE:%.*]] = ptrtoint ptr [[ISA_VALUE]] to i64
  // CHECK: [[SIGNED_VALUE:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[CAST_ISA_VALUE]], i32 2, i64 [[BLENDED_VALUE]])
}

// CHECK-LABEL: define ptr @getTestClassIsa(ptr %t) #0 {
Class getTestClassIsa(TestClass *t) {
  return t->isa;
  // CHECK: [[T_ADDR:%.*]] = alloca ptr, align 8
  // CHECK: [[T:%.*]] = load ptr, ptr [[T_ADDR]], align 8
  // CHECK: [[IVAR:%.*]] = load i32, ptr @"OBJC_IVAR_$_TestClass.isa", align 8
  // CHECK: [[IVAR_CONV:%.*]] = sext i32 [[IVAR]] to i64
  // CHECK: [[ADD_PTR:%.*]] = getelementptr inbounds i8, ptr [[T]], i64 [[IVAR_CONV]]
  // CHECK: [[LOADED_VALUE:%.*]] = load ptr, ptr [[ADD_PTR]], align 8
  // CHECK: [[INT_VALUE:%.*]] = ptrtoint ptr [[ADD_PTR]] to i64
  // CHECK: [[BLENDED_VALUE:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[INT_VALUE]], i64 1234)
  // CHECK: [[NULL_CHECK:%.*]] = icmp ne ptr [[LOADED_VALUE]], null
  // CHECK: [[CAST_VALUE:%.*]] = ptrtoint ptr [[LOADED_VALUE]] to i64
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[CAST_VALUE]], i32 2, i64 [[BLENDED_VALUE]])
}

// Just enough to verify we do actually authenticate qualified Class
// CHECK: define ptr @getTestConstClassIsa(ptr %t) #0 {
Class getTestConstClassIsa(TestConstClass *t) {
  return t->isa;
  // CHECK: [[T_ADDR:%.*]] = alloca ptr, align 8
  // CHECK: [[T:%.*]] = load ptr, ptr [[T_ADDR]], align 8
  // CHECK: [[IVAR:%.*]] = load i32, ptr @"OBJC_IVAR_$_TestConstClass.isa", align 8
  // CHECK: [[IVAR_CONV:%.*]] = sext i32 [[IVAR]] to i64
  // CHECK: [[ADD_PTR:%.*]] = getelementptr inbounds i8, ptr [[T]], i64 [[IVAR_CONV]]
  // CHECK: [[LOADED_VALUE:%.*]] = load ptr, ptr [[ADD_PTR]], align 8
  // CHECK: [[INT_VALUE:%.*]] = ptrtoint ptr [[ADD_PTR]] to i64
  // CHECK: [[BLENDED_VALUE:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[INT_VALUE]], i64 1234)
  // CHECK: [[NULL_CHECK:%.*]] = icmp ne ptr [[LOADED_VALUE]], null
  // CHECK: [[CAST_VALUE:%.*]] = ptrtoint ptr [[LOADED_VALUE]] to i64
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[CAST_VALUE]], i32 2, i64 [[BLENDED_VALUE]])
}

#endif
