// RUN: %clang_cc1 -triple i386-unknown-unknown -fblocks -emit-llvm %s -o - | FileCheck %s

// CHECK: %[[STRUCT_A:.*]] = type { i8 }

struct A
{
    ~A();
};
int foo(A);

void bar(A &a)
{
    // CHECK: call void asm
    asm("" : : "r"(foo(a)) );
    // CHECK: call void @_ZN1AD1Ev
}

namespace TestTemplate {
// Check that the temporary is destructed after the first asm statement.

// CHECK: define {{.*}}void @_ZN12TestTemplate4foo0IvEEvR1A(
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_A]],
// CHECK: %[[CALL:.*]] = call noundef i32 @_Z3foo1A({{.*}}%[[AGG_TMP]])
// CHECK: call void asm sideeffect "", "r,~{dirflag},~{fpsr},~{flags}"(i32 %[[CALL]])
// CHECK: call void @_ZN1AD1Ev({{.*}}%[[AGG_TMP]])
// CHECK: call void asm sideeffect "",

template <class T>
void foo0(A &a) {
  asm("" : : "r"(foo(a)) );
  asm("");
}

void test0(A &a) { foo0<void>(a); }

// Check that the block capture is destructed at the end of the enclosing scope.

// CHECK: define {{.*}}void @_ZN12TestTemplate4foo1IvEEv1A(
// CHECK: %[[BLOCK:.*]] = alloca <{ ptr, i32, i32, ptr, ptr, %[[STRUCT_A]] }>, align 4
// CHECK: %[[BLOCK_CAPTURED:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, %[[STRUCT_A]] }>, ptr %[[BLOCK]], i32 0, i32 5
// CHECK: call void asm sideeffect "", "r,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
// CHECK: call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void @_ZN1AD1Ev({{.*}} %[[BLOCK_CAPTURED]])

template <class T>
void foo1(A a) {
  asm("" : : "r"(^{ (void)a; return 0; }()));
  asm("");
}

void test1(A &a) { foo1<void>(a); }
} // namespace TestTemplate
