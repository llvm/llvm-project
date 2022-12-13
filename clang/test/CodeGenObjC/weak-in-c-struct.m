// RUN: %clang_cc1 -triple arm64-apple-ios11 -fobjc-arc -fblocks -fobjc-runtime=ios-11.0 -emit-llvm -o - %s | FileCheck -check-prefix=ARM64 -check-prefix=COMMON %s
// RUN: %clang_cc1 -triple thumbv7-apple-ios10 -fobjc-arc -fblocks -fobjc-runtime=ios-10.0 -emit-llvm -o - %s | FileCheck -check-prefix=COMMON %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.13 -fobjc-arc -fblocks -fobjc-runtime=macosx-10.13.0 -emit-llvm -o - %s | FileCheck -check-prefix=COMMON %s
// RUN: %clang_cc1 -triple i386-apple-macosx10.13.0 -fobjc-arc -fblocks -fobjc-runtime=macosx-fragile-10.13.0 -emit-llvm -o - %s | FileCheck -check-prefix=COMMON %s

typedef void (^BlockTy)(void);

// COMMON: %[[STRUCT_WEAK:.*]] = type { i32, ptr }

typedef struct {
  int f0;
  __weak id f1;
} Weak;

Weak getWeak(void);
void calleeWeak(Weak);

// ARM64: define{{.*}} void @test_constructor_destructor_Weak()
// ARM64: %[[T:.*]] = alloca %[[STRUCT_WEAK]], align 8
// ARM64: call void @__default_constructor_8_w8(ptr %[[T]])
// ARM64: call void @__destructor_8_w8(ptr %[[T]])
// ARM64: ret void

// ARM64: define linkonce_odr hidden void @__default_constructor_8_w8(ptr noundef %[[DST:.*]])
// ARM64: %[[DST_ADDR:.*]] = alloca ptr, align 8
// ARM64: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// ARM64: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// ARM64: %[[V2:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 8
// ARM64: call void @llvm.memset.p0.i64(ptr align 8 %[[V2]], i8 0, i64 8, i1 false)

// ARM64: define linkonce_odr hidden void @__destructor_8_w8(ptr noundef %[[DST:.*]])
// ARM64: %[[DST_ADDR:.*]] = alloca ptr, align 8
// ARM64: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// ARM64: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// ARM64: %[[V2:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 8
// ARM64: call void @llvm.objc.destroyWeak(ptr %[[V2]])

@interface C
- (void)m:(Weak)a;
@end

void test_constructor_destructor_Weak(void) {
  Weak t;
}

// ARM64: define{{.*}} void @test_copy_constructor_Weak(ptr noundef %{{.*}})
// ARM64: call void @__copy_constructor_8_8_t0w4_w8(ptr %{{.*}}, ptr %{{.*}})
// ARM64: call void @__destructor_8_w8(ptr %{{.*}})

// ARM64: define linkonce_odr hidden void @__copy_constructor_8_8_t0w4_w8(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// ARM64: %[[DST_ADDR:.*]] = alloca ptr, align 8
// ARM64: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// ARM64: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// ARM64: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// ARM64: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// ARM64: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// ARM64: %[[V4:.*]] = load i32, ptr %[[V1]], align 8
// ARM64: store i32 %[[V4]], ptr %[[V0]], align 8
// ARM64: %[[V6:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 8
// ARM64: %[[V9:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 8
// ARM64: call void @llvm.objc.copyWeak(ptr %[[V6]], ptr %[[V9]])

void test_copy_constructor_Weak(Weak *s) {
  Weak t = *s;
}

// ARM64: define{{.*}} void @test_copy_assignment_Weak(ptr noundef %{{.*}}, ptr noundef %{{.*}})
// ARM64: call void @__copy_assignment_8_8_t0w4_w8(ptr %{{.*}}, ptr %{{.*}})

// ARM64: define linkonce_odr hidden void @__copy_assignment_8_8_t0w4_w8(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// ARM64: %[[DST_ADDR:.*]] = alloca ptr, align 8
// ARM64: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// ARM64: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// ARM64: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// ARM64: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// ARM64: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// ARM64: %[[V4:.*]] = load i32, ptr %[[V1]], align 8
// ARM64: store i32 %[[V4]], ptr %[[V0]], align 8
// ARM64: %[[V6:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 8
// ARM64: %[[V9:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 8
// ARM64: %[[V11:.*]] = call ptr @llvm.objc.loadWeakRetained(ptr %[[V9]])
// ARM64: %[[V12:.*]] = call ptr @llvm.objc.storeWeak(ptr %[[V6]], ptr %[[V11]])
// ARM64: call void @llvm.objc.release(ptr %[[V11]])

void test_copy_assignment_Weak(Weak *d, Weak *s) {
  *d = *s;
}

// ARM64: define internal void @__Block_byref_object_copy_(ptr noundef %0, ptr noundef %1)
// ARM64: call void @__move_constructor_8_8_t0w4_w8(ptr %{{.*}}, ptr %{{.*}})

// ARM64: define linkonce_odr hidden void @__move_constructor_8_8_t0w4_w8(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// ARM64: %[[DST_ADDR:.*]] = alloca ptr, align 8
// ARM64: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// ARM64: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// ARM64: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// ARM64: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// ARM64: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// ARM64: %[[V4:.*]] = load i32, ptr %[[V1]], align 8
// ARM64: store i32 %[[V4]], ptr %[[V0]], align 8
// ARM64: %[[V6:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 8
// ARM64: %[[V9:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 8
// ARM64: call void @llvm.objc.moveWeak(ptr %[[V6]], ptr %[[V9]])

void test_move_constructor_Weak(void) {
  __block Weak t;
  BlockTy b = ^{ (void)t; };
}

// ARM64: define{{.*}} void @test_move_assignment_Weak(ptr noundef %{{.*}})
// ARM64: call void @__move_assignment_8_8_t0w4_w8(ptr %{{.*}}, ptr %{{.*}})

// ARM64: define linkonce_odr hidden void @__move_assignment_8_8_t0w4_w8(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// ARM64: %[[DST_ADDR:.*]] = alloca ptr, align 8
// ARM64: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// ARM64: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// ARM64: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// ARM64: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// ARM64: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// ARM64: %[[V4:.*]] = load i32, ptr %[[V1]], align 8
// ARM64: store i32 %[[V4]], ptr %[[V0]], align 8
// ARM64: %[[V6:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 8
// ARM64: %[[V9:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 8
// ARM64: %[[V11:.*]] = call ptr @llvm.objc.loadWeakRetained(ptr %[[V9]])
// ARM64: %[[V12:.*]] = call ptr @llvm.objc.storeWeak(ptr %[[V6]], ptr %[[V11]])
// ARM64: call void @llvm.objc.destroyWeak(ptr %[[V9]])
// ARM64: call void @llvm.objc.release(ptr %[[V11]])

void test_move_assignment_Weak(Weak *p) {
  *p = getWeak();
}

// COMMON: define{{.*}} void @test_parameter_Weak(ptr noundef %[[A:.*]])
// COMMON: call void @__destructor_{{.*}}(ptr %[[A]])

void test_parameter_Weak(Weak a) {
}

// COMMON: define{{.*}} void @test_argument_Weak(ptr noundef %[[A:.*]])
// COMMON: %[[A_ADDR:.*]] = alloca ptr
// COMMON: %[[AGG_TMP:.*]] = alloca %[[STRUCT_WEAK]]
// COMMON: store ptr %[[A]], ptr %[[A_ADDR]]
// COMMON: %[[V0:.*]] = load ptr, ptr %[[A_ADDR]]
// COMMON: call void @__copy_constructor_{{.*}}(ptr %[[AGG_TMP]], ptr %[[V0]])
// COMMON: call void @calleeWeak(ptr noundef %[[AGG_TMP]])
// COMMON-NEXT: ret

void test_argument_Weak(Weak *a) {
  calleeWeak(*a);
}

// COMMON: define{{.*}} void @test_return_Weak(ptr noalias sret(%[[STRUCT_WEAK]]) align {{.*}} %[[AGG_RESULT:.*]], ptr noundef %[[A:.*]])
// COMMON: %[[A_ADDR:.*]] = alloca ptr
// COMMON: store ptr %[[A]], ptr %[[A_ADDR]]
// COMMON: %[[V0:.*]] = load ptr, ptr %[[A_ADDR]]
// COMMON: call void @__copy_constructor_{{.*}}(ptr %[[AGG_RESULT]], ptr %[[V0]])
// COMMON: ret void

Weak test_return_Weak(Weak *a) {
  return *a;
}

// COMMON-LABEL: define{{.*}} void @test_null_receiver(
// COMMON: %[[AGG_TMP:.*]] = alloca %[[STRUCT_WEAK]]
// COMMON: br i1

// COMMON: call void @objc_msgSend({{.*}}, ptr noundef %[[AGG_TMP]])
// COMMON: br

// COMMON: call void @__destructor_{{.*}}(ptr %[[AGG_TMP]])
// COMMON: br

void test_null_receiver(C *c) {
  [c m:getWeak()];
}
