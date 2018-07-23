// RUN: %clang_cc1 -triple x86_64-apple-darwin -fblocks -emit-llvm -o - %s | FileCheck -check-prefix CHECK -check-prefix CHECK-NOARC %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fblocks -emit-llvm -fobjc-arc -o - %s | FileCheck -check-prefix CHECK -check-prefix CHECK-ARC %s

typedef void (^BlockTy)(void);

union U {
  int *i;
  long long *ll;
} __attribute__((transparent_union));

void noescapeFunc0(id, __attribute__((noescape)) BlockTy);
void noescapeFunc1(__attribute__((noescape)) int *);
void noescapeFunc2(__attribute__((noescape)) id);
void noescapeFunc3(__attribute__((noescape)) union U);

// Block descriptors of non-escaping blocks don't need pointers to copy/dispose
// helper functions.

// CHECK: %[[STRUCT_BLOCK_DESCRIPTOR:.*]] = type { i64, i64 }
// CHECK: @[[BLOCK_DESCIPTOR_TMP_2:.*]] = internal constant { i64, i64, i8*, i64 } { i64 0, i64 40, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @{{.*}}, i32 0, i32 0), i64 256 }, align 8

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
// CHECK-ARC: call void @objc_storeStrong(i8** %[[B_ADDR]], i8* %[[B]])
// CHECK-ARC: %[[V0:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>* %[[BLOCK]], i32 0, i32 5
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
// CHECK-ARC: %[[V3:.*]] = call i8* @objc_retain(i8* %[[V2]]) #3
// CHECK-ARC: store i8* %[[V3]], i8** %[[BLOCK_CAPTURED]], align 8
// CHECK: call void @noescapeFunc0(
// CHECK-ARC: call void @objc_storeStrong(i8** %[[V0]], i8* null)
// CHECK-ARC: call void @objc_storeStrong(i8** %[[B_ADDR]], i8* null)

// Non-escaping blocks don't need copy/dispose helper functions.

// CHECK-NOT: define internal void @__copy_helper_block_
// CHECK-NOT: define internal void @__destroy_helper_block_

void func(id);

void test6(id a, id b) {
  noescapeFunc0(a, ^{ func(b); });
}
