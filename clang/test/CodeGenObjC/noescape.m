// RUN: %clang_cc1 -triple x86_64-apple-darwin -fblocks -emit-llvm -o - %s | FileCheck %s

typedef void (^BlockTy)(void);

union U {
  int *i;
  long long *ll;
} __attribute__((transparent_union));

void noescapeFunc0(id, __attribute__((noescape)) BlockTy);
void noescapeFunc1(__attribute__((noescape)) int *);
void noescapeFunc2(__attribute__((noescape)) id);
void noescapeFunc3(__attribute__((noescape)) union U);

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
