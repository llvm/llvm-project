// RUN: %clang_cc1 -no-opaque-pointers -triple arm64-apple-ios11 -fobjc-arc -fblocks -fobjc-runtime=ios-11.0 -emit-llvm -o - %s | FileCheck -check-prefix=ARM64 -check-prefix=COMMON %s
// RUN: %clang_cc1 -no-opaque-pointers -triple thumbv7-apple-ios10 -fobjc-arc -fblocks -fobjc-runtime=ios-10.0 -emit-llvm -o - %s | FileCheck -check-prefix=COMMON %s
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-macosx10.13 -fobjc-arc -fblocks -fobjc-runtime=macosx-10.13.0 -emit-llvm -o - %s | FileCheck -check-prefix=COMMON %s
// RUN: %clang_cc1 -no-opaque-pointers -triple i386-apple-macosx10.13.0 -fobjc-arc -fblocks -fobjc-runtime=macosx-fragile-10.13.0 -emit-llvm -o - %s | FileCheck -check-prefix=COMMON %s

typedef void (^BlockTy)(void);

// COMMON: %[[STRUCT_WEAK:.*]] = type { i32, i8* }

typedef struct {
  int f0;
  __weak id f1;
} Weak;

Weak getWeak(void);
void calleeWeak(Weak);

// ARM64: define{{.*}} void @test_constructor_destructor_Weak()
// ARM64: %[[T:.*]] = alloca %[[STRUCT_WEAK]], align 8
// ARM64: %[[V0:.*]] = bitcast %[[STRUCT_WEAK]]* %[[T]] to i8**
// ARM64: call void @__default_constructor_8_w8(i8** %[[V0]])
// ARM64: %[[V1:.*]] = bitcast %[[STRUCT_WEAK]]* %[[T]] to i8**
// ARM64: call void @__destructor_8_w8(i8** %[[V1]])
// ARM64: ret void

// ARM64: define linkonce_odr hidden void @__default_constructor_8_w8(i8** noundef %[[DST:.*]])
// ARM64: %[[DST_ADDR:.*]] = alloca i8**, align 8
// ARM64: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// ARM64: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// ARM64: %[[V1]] = bitcast i8** %[[V0]] to i8*
// ARM64: %[[V2:.*]] = getelementptr inbounds i8, i8* %[[V1]], i64 8
// ARM64: %[[V3:.*]] = bitcast i8* %[[V2]] to i8**
// ARM64: %[[V4:.*]] = bitcast i8** %[[V3]] to i8*
// ARM64: call void @llvm.memset.p0i8.i64(i8* align 8 %[[V4]], i8 0, i64 8, i1 false)

// ARM64: define linkonce_odr hidden void @__destructor_8_w8(i8** noundef %[[DST:.*]])
// ARM64: %[[DST_ADDR:.*]] = alloca i8**, align 8
// ARM64: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// ARM64: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// ARM64: %[[V1:.*]] = bitcast i8** %[[V0]] to i8*
// ARM64: %[[V2:.*]] = getelementptr inbounds i8, i8* %[[V1]], i64 8
// ARM64: %[[V3:.*]] = bitcast i8* %[[V2]] to i8**
// ARM64: call void @llvm.objc.destroyWeak(i8** %[[V3]])

@interface C
- (void)m:(Weak)a;
@end

void test_constructor_destructor_Weak(void) {
  Weak t;
}

// ARM64: define{{.*}} void @test_copy_constructor_Weak(%[[STRUCT_WEAK]]* noundef %{{.*}})
// ARM64: call void @__copy_constructor_8_8_t0w4_w8(i8** %{{.*}}, i8** %{{.*}})
// ARM64: call void @__destructor_8_w8(i8** %{{.*}})

// ARM64: define linkonce_odr hidden void @__copy_constructor_8_8_t0w4_w8(i8** noundef %[[DST:.*]], i8** noundef %[[SRC:.*]])
// ARM64: %[[DST_ADDR:.*]] = alloca i8**, align 8
// ARM64: %[[SRC_ADDR:.*]] = alloca i8**, align 8
// ARM64: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// ARM64: store i8** %[[SRC]], i8*** %[[SRC_ADDR]], align 8
// ARM64: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// ARM64: %[[V1:.*]] = load i8**, i8*** %[[SRC_ADDR]], align 8
// ARM64: %[[V2:.*]] = bitcast i8** %[[V0]] to i32*
// ARM64: %[[V3:.*]] = bitcast i8** %[[V1]] to i32*
// ARM64: %[[V4:.*]] = load i32, i32* %[[V3]], align 8
// ARM64: store i32 %[[V4]], i32* %[[V2]], align 8
// ARM64: %[[V5:.*]] = bitcast i8** %[[V0]] to i8*
// ARM64: %[[V6:.*]] = getelementptr inbounds i8, i8* %[[V5]], i64 8
// ARM64: %[[V7:.*]] = bitcast i8* %[[V6]] to i8**
// ARM64: %[[V8:.*]] = bitcast i8** %[[V1]] to i8*
// ARM64: %[[V9:.*]] = getelementptr inbounds i8, i8* %[[V8]], i64 8
// ARM64: %[[V10:.*]] = bitcast i8* %[[V9]] to i8**
// ARM64: call void @llvm.objc.copyWeak(i8** %[[V7]], i8** %[[V10]])

void test_copy_constructor_Weak(Weak *s) {
  Weak t = *s;
}

// ARM64: define{{.*}} void @test_copy_assignment_Weak(%[[STRUCT_WEAK]]* noundef %{{.*}}, %[[STRUCT_WEAK]]* noundef %{{.*}})
// ARM64: call void @__copy_assignment_8_8_t0w4_w8(i8** %{{.*}}, i8** %{{.*}})

// ARM64: define linkonce_odr hidden void @__copy_assignment_8_8_t0w4_w8(i8** noundef %[[DST:.*]], i8** noundef %[[SRC:.*]])
// ARM64: %[[DST_ADDR:.*]] = alloca i8**, align 8
// ARM64: %[[SRC_ADDR:.*]] = alloca i8**, align 8
// ARM64: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// ARM64: store i8** %[[SRC]], i8*** %[[SRC_ADDR]], align 8
// ARM64: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// ARM64: %[[V1:.*]] = load i8**, i8*** %[[SRC_ADDR]], align 8
// ARM64: %[[V2:.*]] = bitcast i8** %[[V0]] to i32*
// ARM64: %[[V3:.*]] = bitcast i8** %[[V1]] to i32*
// ARM64: %[[V4:.*]] = load i32, i32* %[[V3]], align 8
// ARM64: store i32 %[[V4]], i32* %[[V2]], align 8
// ARM64: %[[V5:.*]] = bitcast i8** %[[V0]] to i8*
// ARM64: %[[V6:.*]] = getelementptr inbounds i8, i8* %[[V5]], i64 8
// ARM64: %[[V7:.*]] = bitcast i8* %[[V6]] to i8**
// ARM64: %[[V8:.*]] = bitcast i8** %[[V1]] to i8*
// ARM64: %[[V9:.*]] = getelementptr inbounds i8, i8* %[[V8]], i64 8
// ARM64: %[[V10:.*]] = bitcast i8* %[[V9]] to i8**
// ARM64: %[[V11:.*]] = call i8* @llvm.objc.loadWeakRetained(i8** %[[V10]])
// ARM64: %[[V12:.*]] = call i8* @llvm.objc.storeWeak(i8** %[[V7]], i8* %[[V11]])
// ARM64: call void @llvm.objc.release(i8* %[[V11]])

void test_copy_assignment_Weak(Weak *d, Weak *s) {
  *d = *s;
}

// ARM64: define internal void @__Block_byref_object_copy_(i8* noundef %0, i8* noundef %1)
// ARM64: call void @__move_constructor_8_8_t0w4_w8(i8** %{{.*}}, i8** %{{.*}})

// ARM64: define linkonce_odr hidden void @__move_constructor_8_8_t0w4_w8(i8** noundef %[[DST:.*]], i8** noundef %[[SRC:.*]])
// ARM64: %[[DST_ADDR:.*]] = alloca i8**, align 8
// ARM64: %[[SRC_ADDR:.*]] = alloca i8**, align 8
// ARM64: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// ARM64: store i8** %[[SRC]], i8*** %[[SRC_ADDR]], align 8
// ARM64: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// ARM64: %[[V1:.*]] = load i8**, i8*** %[[SRC_ADDR]], align 8
// ARM64: %[[V2:.*]] = bitcast i8** %[[V0]] to i32*
// ARM64: %[[V3:.*]] = bitcast i8** %[[V1]] to i32*
// ARM64: %[[V4:.*]] = load i32, i32* %[[V3]], align 8
// ARM64: store i32 %[[V4]], i32* %[[V2]], align 8
// ARM64: %[[V5:.*]] = bitcast i8** %[[V0]] to i8*
// ARM64: %[[V6:.*]] = getelementptr inbounds i8, i8* %[[V5]], i64 8
// ARM64: %[[V7:.*]] = bitcast i8* %[[V6]] to i8**
// ARM64: %[[V8:.*]] = bitcast i8** %[[V1]] to i8*
// ARM64: %[[V9:.*]] = getelementptr inbounds i8, i8* %[[V8]], i64 8
// ARM64: %[[V10:.*]] = bitcast i8* %[[V9]] to i8**
// ARM64: call void @llvm.objc.moveWeak(i8** %[[V7]], i8** %[[V10]])

void test_move_constructor_Weak(void) {
  __block Weak t;
  BlockTy b = ^{ (void)t; };
}

// ARM64: define{{.*}} void @test_move_assignment_Weak(%[[STRUCT_WEAK]]* noundef %{{.*}})
// ARM64: call void @__move_assignment_8_8_t0w4_w8(i8** %{{.*}}, i8** %{{.*}})

// ARM64: define linkonce_odr hidden void @__move_assignment_8_8_t0w4_w8(i8** noundef %[[DST:.*]], i8** noundef %[[SRC:.*]])
// ARM64: %[[DST_ADDR:.*]] = alloca i8**, align 8
// ARM64: %[[SRC_ADDR:.*]] = alloca i8**, align 8
// ARM64: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// ARM64: store i8** %[[SRC]], i8*** %[[SRC_ADDR]], align 8
// ARM64: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// ARM64: %[[V1:.*]] = load i8**, i8*** %[[SRC_ADDR]], align 8
// ARM64: %[[V2:.*]] = bitcast i8** %[[V0]] to i32*
// ARM64: %[[V3:.*]] = bitcast i8** %[[V1]] to i32*
// ARM64: %[[V4:.*]] = load i32, i32* %[[V3]], align 8
// ARM64: store i32 %[[V4]], i32* %[[V2]], align 8
// ARM64: %[[V5:.*]] = bitcast i8** %[[V0]] to i8*
// ARM64: %[[V6:.*]] = getelementptr inbounds i8, i8* %[[V5]], i64 8
// ARM64: %[[V7:.*]] = bitcast i8* %[[V6]] to i8**
// ARM64: %[[V8:.*]] = bitcast i8** %[[V1]] to i8*
// ARM64: %[[V9:.*]] = getelementptr inbounds i8, i8* %[[V8]], i64 8
// ARM64: %[[V10:.*]] = bitcast i8* %[[V9]] to i8**
// ARM64: %[[V11:.*]] = call i8* @llvm.objc.loadWeakRetained(i8** %[[V10]])
// ARM64: %[[V12:.*]] = call i8* @llvm.objc.storeWeak(i8** %[[V7]], i8* %[[V11]])
// ARM64: call void @llvm.objc.destroyWeak(i8** %[[V10]])
// ARM64: call void @llvm.objc.release(i8* %[[V11]])

void test_move_assignment_Weak(Weak *p) {
  *p = getWeak();
}

// COMMON: define{{.*}} void @test_parameter_Weak(%[[STRUCT_WEAK]]* noundef %[[A:.*]])
// COMMON: %[[V0:.*]] = bitcast %[[STRUCT_WEAK]]* %[[A]] to i8**
// COMMON: call void @__destructor_{{.*}}(i8** %[[V0]])

void test_parameter_Weak(Weak a) {
}

// COMMON: define{{.*}} void @test_argument_Weak(%[[STRUCT_WEAK]]* noundef %[[A:.*]])
// COMMON: %[[A_ADDR:.*]] = alloca %[[STRUCT_WEAK]]*
// COMMON: %[[AGG_TMP:.*]] = alloca %[[STRUCT_WEAK]]
// COMMON: store %[[STRUCT_WEAK]]* %[[A]], %[[STRUCT_WEAK]]** %[[A_ADDR]]
// COMMON: %[[V0:.*]] = load %[[STRUCT_WEAK]]*, %[[STRUCT_WEAK]]** %[[A_ADDR]]
// COMMON: %[[V1:.*]] = bitcast %[[STRUCT_WEAK]]* %[[AGG_TMP]] to i8**
// COMMON: %[[V2:.*]] = bitcast %[[STRUCT_WEAK]]* %[[V0]] to i8**
// COMMON: call void @__copy_constructor_{{.*}}(i8** %[[V1]], i8** %[[V2]])
// COMMON: call void @calleeWeak(%[[STRUCT_WEAK]]* noundef %[[AGG_TMP]])
// COMMON-NEXT: ret

void test_argument_Weak(Weak *a) {
  calleeWeak(*a);
}

// COMMON: define{{.*}} void @test_return_Weak(%[[STRUCT_WEAK]]* noalias sret(%[[STRUCT_WEAK]]) align {{.*}} %[[AGG_RESULT:.*]], %[[STRUCT_WEAK]]* noundef %[[A:.*]])
// COMMON: %[[A_ADDR:.*]] = alloca %[[STRUCT_WEAK]]*
// COMMON: store %[[STRUCT_WEAK]]* %[[A]], %[[STRUCT_WEAK]]** %[[A_ADDR]]
// COMMON: %[[V0:.*]] = load %[[STRUCT_WEAK]]*, %[[STRUCT_WEAK]]** %[[A_ADDR]]
// COMMON: %[[V1:.*]] = bitcast %[[STRUCT_WEAK]]* %[[AGG_RESULT]] to i8**
// COMMON: %[[V2:.*]] = bitcast %[[STRUCT_WEAK]]* %[[V0]] to i8**
// COMMON: call void @__copy_constructor_{{.*}}(i8** %[[V1]], i8** %[[V2]])
// COMMON: ret void

Weak test_return_Weak(Weak *a) {
  return *a;
}

// COMMON-LABEL: define{{.*}} void @test_null_receiver(
// COMMON: %[[AGG_TMP:.*]] = alloca %[[STRUCT_WEAK]]
// COMMON: br i1

// COMMON: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %[[STRUCT_WEAK]]*)*)({{.*}}, %[[STRUCT_WEAK]]* noundef %[[AGG_TMP]])
// COMMON: br

// COMMON: %[[V6:.*]] = bitcast %[[STRUCT_WEAK]]* %[[AGG_TMP]] to i8**
// COMMON: call void @__destructor_{{.*}}(i8** %[[V6]])
// COMMON: br

void test_null_receiver(C *c) {
  [c m:getWeak()];
}
