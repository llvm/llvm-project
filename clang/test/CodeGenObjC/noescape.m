// RUN: %clang_cc1 -triple x86_64-apple-darwin -fblocks -emit-llvm -o - %s | FileCheck -check-prefix CHECK -check-prefix CHECK-NOARC %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fblocks -emit-llvm -fobjc-arc -o - %s | FileCheck -check-prefix CHECK -check-prefix CHECK-ARC %s

typedef void (^BlockTy)(void);

union U {
  int *i;
  long long *ll;
} __attribute__((transparent_union));

void escapingFunc0(BlockTy);
void noescapeFunc0(id, __attribute__((noescape)) BlockTy);
void noescapeFunc1(__attribute__((noescape)) int *);
void noescapeFunc2(__attribute__((noescape)) id);
void noescapeFunc3(__attribute__((noescape)) union U);

// Block descriptors of non-escaping blocks don't need pointers to copy/dispose
// helper functions.

// CHECK: %[[STRUCT_BLOCK_DESCRIPTOR:.*]] = type { i64, i64 }

// When the block is non-escaping, copy/dispose helpers aren't generated, so the
// block layout string must include information about __strong captures.

// CHECK-NOARC: %[[STRUCT_BLOCK_BYREF_B0:.*]] = type { i8*, %[[STRUCT_BLOCK_BYREF_B0]]*, i32, i32, i8*, %[[STRUCT_S0:.*]] }
// CHECK-ARC: %[[STRUCT_BLOCK_BYREF_B0:.*]] = type { i8*, %[[STRUCT_BLOCK_BYREF_B0]]*, i32, i32, i8*, i8*, i8*, %[[STRUCT_S0:.*]] }
// CHECK: %[[STRUCT_S0]] = type { i8*, i8* }
// CHECK: @[[BLOCK_DESCIPTOR_TMP_2:.*ls32l8"]] = linkonce_odr hidden unnamed_addr constant { i64, i64, i8*, i64 } { i64 0, i64 40, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @{{.*}}, i32 0, i32 0), i64 256 }, align 8

// CHECK-LABEL: define void @test0(
// CHECK: call void @noescapeFunc0({{.*}}, {{.*}} nocapture {{.*}})
// CHECK: declare void @noescapeFunc0(i8*, {{.*}} nocapture)
void test0(BlockTy b) {
  noescapeFunc0(0, b);
}

// CHECK-LABEL: define void @test1(
// CHECK: call void @noescapeFunc1({{.*}} nocapture {{.*}})
// CHECK: declare void @noescapeFunc1({{.*}} nocapture)
void test1(int *i) {
  noescapeFunc1(i);
}

// CHECK-LABEL: define void @test2(
// CHECK: call void @noescapeFunc2({{.*}} nocapture {{.*}})
// CHECK: declare void @noescapeFunc2({{.*}} nocapture)
void test2(id i) {
  noescapeFunc2(i);
}

// CHECK-LABEL: define void @test3(
// CHECK: call void @noescapeFunc3({{.*}} nocapture {{.*}})
// CHECK: declare void @noescapeFunc3({{.*}} nocapture)
void test3(union U u) {
  noescapeFunc3(u);
}

// CHECK: define internal void @"\01-[C0 m0:]"({{.*}}, {{.*}}, {{.*}} nocapture {{.*}})

// CHECK-LABEL: define void @test4(
// CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i32*)*)(i8* {{.*}}, i8* {{.*}}, i32* nocapture {{.*}})

@interface C0
-(void) m0:(int*)__attribute__((noescape)) p0;
@end

@implementation C0
-(void) m0:(int*)__attribute__((noescape)) p0 {
}
@end

void test4(C0 *c0, int *p) {
  [c0 m0:p];
}

// CHECK-LABEL: define void @test5(
// CHECK: call void {{.*}}(i8* bitcast ({ i8**, i32, i32, i8*, {{.*}} }* @{{.*}} to i8*), i32* nocapture {{.*}})
// CHECK: call void {{.*}}(i8* {{.*}}, i32* nocapture {{.*}})
// CHECK: define internal void @{{.*}}(i8* {{.*}}, i32* nocapture {{.*}})

typedef void (^BlockTy2)(__attribute__((noescape)) int *);

void test5(BlockTy2 b, int *p) {
  ^(int *__attribute__((noescape)) p0){}(p);
  b(p);
}

// If the block is non-escaping, set the BLOCK_IS_NOESCAPE and BLOCK_IS_GLOBAL
// bits of field 'flags' and set the 'isa' field to 'NSConcreteGlobalBlock'.

// CHECK: define void @test6(i8* %{{.*}}, i8* %[[B:.*]])
// CHECK: %{{.*}} = alloca i8*, align 8
// CHECK: %[[B_ADDR:.*]] = alloca i8*, align 8
// CHECK: %[[BLOCK:.*]] = alloca <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, align 8
// CHECK-NOARC: store i8* %[[B]], i8** %[[B_ADDR]], align 8
// CHECK-ARC: store i8* null, i8** %[[B_ADDR]], align 8
// CHECK-ARC: call void @llvm.objc.storeStrong(i8** %[[B_ADDR]], i8* %[[B]])
// CHECK: %[[BLOCK_ISA:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>* %[[BLOCK]], i32 0, i32 0
// CHECK: store i8* bitcast (i8** @_NSConcreteGlobalBlock to i8*), i8** %[[BLOCK_ISA]], align 8
// CHECK: %[[BLOCK_FLAGS:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>* %[[BLOCK]], i32 0, i32 1
// CHECK: store i32 -796917760, i32* %[[BLOCK_FLAGS]], align 8
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>* %[[BLOCK]], i32 0, i32 4
// CHECK: store %[[STRUCT_BLOCK_DESCRIPTOR]]* bitcast ({ i64, i64, i8*, i64 }* @[[BLOCK_DESCIPTOR_TMP_2]] to %[[STRUCT_BLOCK_DESCRIPTOR]]*), %[[STRUCT_BLOCK_DESCRIPTOR]]** %[[BLOCK_DESCRIPTOR]], align 8
// CHECK: %[[BLOCK_CAPTURED:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>* %[[BLOCK]], i32 0, i32 5
// CHECK-NOARC: %[[V1:.*]] = load i8*, i8** %[[B_ADDR]], align 8
// CHECK-NOARC: store i8* %[[V1]], i8** %[[BLOCK_CAPTURED]], align 8
// CHECK-ARC: %[[V2:.*]] = load i8*, i8** %[[B_ADDR]], align 8
// CHECK-ARC: %[[V3:.*]] = call i8* @llvm.objc.retain(i8* %[[V2]])
// CHECK-ARC: store i8* %[[V3]], i8** %[[BLOCK_CAPTURED]], align 8
// CHECK: call void @noescapeFunc0(
// CHECK-ARC: call void @llvm.objc.storeStrong(i8** %[[BLOCK_CAPTURED]], i8* null)
// CHECK-ARC: call void @llvm.objc.storeStrong(i8** %[[B_ADDR]], i8* null)

// Non-escaping blocks don't need copy/dispose helper functions.

// CHECK-NOT: define internal void @__copy_helper_block_
// CHECK-NOT: define internal void @__destroy_helper_block_

void func(id);

void test6(id a, id b) {
  noescapeFunc0(a, ^{ func(b); });
}

// We don't need either the byref helper functions or the byref structs for
// __block variables that are not captured by escaping blocks.

// CHECK: define void @test7(
// CHECK: alloca i8*, align 8
// CHECK: %[[B0:.*]] = alloca i8*, align 8
// CHECK: %[[BLOCK:.*]] = alloca <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8** }>, align 8
// CHECK: %[[BLOCK_CAPTURED:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8** }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8** }>* %[[BLOCK]], i32 0, i32 5
// CHECK: store i8** %[[B0]], i8*** %[[BLOCK_CAPTURED]], align 8

// CHECK-ARC-NOT: define internal void @__Block_byref_object_copy_
// CHECK-ARC-NOT: define internal void @__Block_byref_object_dispose_

void test7() {
  id a;
  __block id b0;
  noescapeFunc0(a, ^{ (void)b0; });
}

// __block variables captured by escaping blocks need byref helper functions.

// CHECK: define void @test8(
// CHECK: %[[A:.*]] = alloca i8*, align 8
// CHECK: %[[B0:.*]] = alloca %[[STRUCT_BLOCK_BYREF_B0]], align 8
// CHECK: alloca <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, align 8
// CHECK: %[[BLOCK1:.*]] = alloca <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, align 8
// CHECK: %[[BLOCK_CAPTURED7:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>* %[[BLOCK1]], i32 0, i32 5
// CHECK: %[[V3:.*]] = bitcast %[[STRUCT_BLOCK_BYREF_B0]]* %[[B0]] to i8*
// CHECK: store i8* %[[V3]], i8** %[[BLOCK_CAPTURED7]], align 8

// CHECK-ARC: define internal void @__Block_byref_object_copy_
// CHECK-ARC: define internal void @__Block_byref_object_dispose_
// CHECK: define linkonce_odr hidden void @__copy_helper_block_
// CHECK: define linkonce_odr hidden void @__destroy_helper_block_

struct S0 {
  id a, b;
};

void test8() {
  id a;
  __block struct S0 b0;
  noescapeFunc0(a, ^{ (void)b0; });
  escapingFunc0(^{ (void)b0; });
}
