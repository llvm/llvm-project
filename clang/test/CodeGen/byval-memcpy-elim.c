// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin10 < %s | FileCheck %s

struct Test1S {
 long NumDecls;
 long X;
 long Y;
};
struct Test2S {
 long NumDecls;
 long X;
};

// Make sure we don't generate extra memcpy for lvalues
void test1a(struct Test1S, struct Test2S);
// CHECK-LABEL: define{{.*}} void @test1(
// CHECK-NOT: memcpy
// CHECK: call void @test1a
void test1(struct Test1S *A, struct Test2S *B) {
  test1a(*A, *B);
}

// The above gets tricker when the byval argument requires higher alignment
// than the natural alignment of the type in question.

// Make sure we do generate a memcpy when we cannot guarantee alignment.
struct Test3S {
  int a,b,c,d,e,f,g,h,i,j,k,l;
};
void test2a(struct Test3S q);
// CHECK-LABEL: define{{.*}} void @test2(
// CHECK: alloca %struct.Test3S, align 8
// CHECK: memcpy
// CHECK: call void @test2a
void test2(struct Test3S *q) {
  test2a(*q);
}

// But make sure we don't generate a memcpy when we can guarantee alignment.
void fooey(void);
// CHECK-LABEL: define{{.*}} void @test3(
// CHECK: alloca %struct.Test3S, align 8
// CHECK: call void @fooey
// CHECK-NOT: memcpy
// CHECK: call void @test2a
// CHECK-NOT: memcpy
// CHECK: call void @test2a
void test3(struct Test3S a) {
  struct Test3S b = a;
  fooey();
  test2a(a);
  test2a(b);
}
