// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
struct A { 
  void f(); 
  
  int a;
};

struct B : A { 
  double b;
};

void f() {
  B b;
  
  b.f();
}

// CHECK: define{{.*}} ptr @_Z1fP1A(ptr noundef %a) [[NUW:#[0-9]+]]
B *f(A *a) {
  // CHECK-NOT: br label
  // CHECK: ret ptr
  return static_cast<B*>(a);
}

// PR5965
namespace PR5965 {

// CHECK: define{{.*}} ptr @_ZN6PR59651fEP1B(ptr noundef %b) [[NUW]]
A *f(B* b) {
  // CHECK-NOT: br label
  // CHECK: ret ptr
  return b;
}

}

// Don't crash on a derived-to-base conversion of an r-value
// aggregate.
namespace test3 {
  struct A {};
  struct B : A {};

  void foo(A a);
  void test() {
    foo(B());
  }
}

// Ensure volatile is preserved during derived-to-base conversion. 
namespace PR127683 {

struct Base {
  int Val;
};
  
struct Derived : Base { };
  
volatile Derived Obj;

// CHECK-LABEL: define void @_ZN8PR12768319test_volatile_storeEv()
// CHECK:         store volatile i32 0, ptr @_ZN8PR1276833ObjE, align 4
void test_volatile_store() {
  Obj.Val = 0;
}

// CHECK-LABEL: define void @_ZN8PR12768318test_volatile_loadEv()
// CHECK:         %0 = load volatile i32, ptr @_ZN8PR1276833ObjE, align 4
void test_volatile_load() {
  [[maybe_unused]] int Val = Obj.Val;
}

}

// CHECK: attributes [[NUW]] = { mustprogress noinline nounwind{{.*}} }
