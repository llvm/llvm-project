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

// When the block is non-escaping, copy/dispose helpers aren't generated, so the
// block layout string must include information about __strong captures.

// CHECK-NOARC: %[[STRUCT_BLOCK_BYREF_B0:.*]] = type { ptr, ptr, i32, i32, ptr, %[[STRUCT_S0:.*]] }
// CHECK-ARC: %[[STRUCT_BLOCK_BYREF_B0:.*]] = type { ptr, ptr, i32, i32, ptr, ptr, ptr, %[[STRUCT_S0:.*]] }
// CHECK: %[[STRUCT_S0]] = type { ptr, ptr }
// CHECK: @[[BLOCK_DESCIPTOR_TMP_2:.*ls32l8"]] = linkonce_odr hidden unnamed_addr constant { i64, i64, ptr, i64 } { i64 0, i64 40, ptr @{{.*}}, i64 256 }, align 8

// CHECK-LABEL: define{{.*}} void @test0(
// CHECK: call void @noescapeFunc0({{.*}}, {{.*}} nocapture {{.*}})
// CHECK: declare void @noescapeFunc0(ptr noundef, {{.*}} nocapture noundef)
void test0(BlockTy b) {
  noescapeFunc0(0, b);
}

// CHECK-LABEL: define{{.*}} void @test1(
// CHECK: call void @noescapeFunc1({{.*}} nocapture {{.*}})
// CHECK: declare void @noescapeFunc1({{.*}} nocapture noundef)
void test1(int *i) {
  noescapeFunc1(i);
}

// CHECK-LABEL: define{{.*}} void @test2(
// CHECK: call void @noescapeFunc2({{.*}} nocapture {{.*}})
// CHECK: declare void @noescapeFunc2({{.*}} nocapture noundef)
void test2(id i) {
  noescapeFunc2(i);
}

// CHECK-LABEL: define{{.*}} void @test3(
// CHECK: call void @noescapeFunc3({{.*}} nocapture {{.*}})
// CHECK: declare void @noescapeFunc3({{.*}} nocapture)
void test3(union U u) {
  noescapeFunc3(u);
}

// CHECK: define internal void @"\01-[C0 m0:]"({{.*}}, {{.*}}, {{.*}} nocapture {{.*}})

// CHECK-LABEL: define{{.*}} void @test4(
// CHECK: call void @objc_msgSend(ptr {{.*}}, ptr {{.*}}, ptr nocapture {{.*}})

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

// CHECK-LABEL: define{{.*}} void @test5(
// CHECK: call void {{.*}}(ptr noundef @{{.*}}, ptr nocapture {{.*}})
// CHECK: call void {{.*}}(ptr {{.*}}, ptr nocapture {{.*}})
// CHECK: define internal void @{{.*}}(ptr {{.*}}, ptr nocapture {{.*}})

typedef void (^BlockTy2)(__attribute__((noescape)) int *);

void test5(BlockTy2 b, int *p) {
  ^(int *__attribute__((noescape)) p0){}(p);
  b(p);
}

// If the block is non-escaping, set the BLOCK_IS_NOESCAPE and BLOCK_IS_GLOBAL
// bits of field 'flags' and set the 'isa' field to 'NSConcreteGlobalBlock'.

// CHECK: define{{.*}} void @test6(ptr noundef %{{.*}}, ptr noundef %[[B:.*]])
// CHECK: %{{.*}} = alloca ptr, align 8
// CHECK: %[[B_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[BLOCK:.*]] = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
// CHECK-NOARC: store ptr %[[B]], ptr %[[B_ADDR]], align 8
// CHECK-ARC: store ptr null, ptr %[[B_ADDR]], align 8
// CHECK-ARC: call void @llvm.objc.storeStrong(ptr %[[B_ADDR]], ptr %[[B]])
// CHECK: %[[BLOCK_ISA:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %[[BLOCK]], i32 0, i32 0
// CHECK: store ptr @_NSConcreteGlobalBlock, ptr %[[BLOCK_ISA]], align 8
// CHECK: %[[BLOCK_FLAGS:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %[[BLOCK]], i32 0, i32 1
// CHECK: store i32 -796917760, ptr %[[BLOCK_FLAGS]], align 8
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %[[BLOCK]], i32 0, i32 4
// CHECK: store ptr @[[BLOCK_DESCIPTOR_TMP_2]], ptr %[[BLOCK_DESCRIPTOR]], align 8
// CHECK: %[[BLOCK_CAPTURED:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %[[BLOCK]], i32 0, i32 5
// CHECK-NOARC: %[[V1:.*]] = load ptr, ptr %[[B_ADDR]], align 8
// CHECK-NOARC: store ptr %[[V1]], ptr %[[BLOCK_CAPTURED]], align 8
// CHECK-ARC: %[[V2:.*]] = load ptr, ptr %[[B_ADDR]], align 8
// CHECK-ARC: %[[V3:.*]] = call ptr @llvm.objc.retain(ptr %[[V2]])
// CHECK-ARC: store ptr %[[V3]], ptr %[[BLOCK_CAPTURED]], align 8
// CHECK: call void @noescapeFunc0(
// CHECK-ARC: call void @llvm.objc.storeStrong(ptr %[[BLOCK_CAPTURED]], ptr null)
// CHECK-ARC: call void @llvm.objc.storeStrong(ptr %[[B_ADDR]], ptr null)

// Non-escaping blocks don't need copy/dispose helper functions.

// CHECK-NOT: define internal void @__copy_helper_block_
// CHECK-NOT: define internal void @__destroy_helper_block_

void func(id);

void test6(id a, id b) {
  noescapeFunc0(a, ^{ func(b); });
}

// We don't need either the byref helper functions or the byref structs for
// __block variables that are not captured by escaping blocks.

// CHECK: define{{.*}} void @test7(
// CHECK: alloca ptr, align 8
// CHECK: %[[B0:.*]] = alloca ptr, align 8
// CHECK: %[[BLOCK:.*]] = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
// CHECK: %[[BLOCK_CAPTURED:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %[[BLOCK]], i32 0, i32 5
// CHECK: store ptr %[[B0]], ptr %[[BLOCK_CAPTURED]], align 8

// CHECK-ARC-NOT: define internal void @__Block_byref_object_copy_
// CHECK-ARC-NOT: define internal void @__Block_byref_object_dispose_

void test7(void) {
  id a;
  __block id b0;
  noescapeFunc0(a, ^{ (void)b0; });
}

// __block variables captured by escaping blocks need byref helper functions.

// CHECK: define{{.*}} void @test8(
// CHECK: %[[A:.*]] = alloca ptr, align 8
// CHECK: %[[B0:.*]] = alloca %[[STRUCT_BLOCK_BYREF_B0]], align 8
// CHECK: alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
// CHECK: %[[BLOCK1:.*]] = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
// CHECK: %[[BLOCK_CAPTURED7:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %[[BLOCK1]], i32 0, i32 5
// CHECK: store ptr %[[B0]], ptr %[[BLOCK_CAPTURED7]], align 8

// CHECK-ARC: define internal void @__Block_byref_object_copy_
// CHECK-ARC: define internal void @__Block_byref_object_dispose_
// CHECK: define linkonce_odr hidden void @__copy_helper_block_
// CHECK: define linkonce_odr hidden void @__destroy_helper_block_

struct S0 {
  id a, b;
};

void test8(void) {
  id a;
  __block struct S0 b0;
  noescapeFunc0(a, ^{ (void)b0; });
  escapingFunc0(^{ (void)b0; });
}
