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
  mutable int MutVal;
};
  
struct Derived : Base { };
  
volatile Derived Obj;
volatile Derived *Ptr;
Derived *volatile VolPtr;
volatile Derived *volatile VPtrToV;

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

// CHECK-LABEL: define void @_ZN8PR12768323test_volatile_ptr_storeEv()
// CHECK:         store volatile i32 0, ptr %{{.*}}, align 4
void test_volatile_ptr_store() {
  Ptr->Val = 0;
}

// CHECK-LABEL: define void @_ZN8PR12768322test_volatile_ptr_loadEv()
// CHECK:         load volatile i32, ptr %{{.*}}, align 4
void test_volatile_ptr_load() {
  [[maybe_unused]] int Val = Ptr->Val;
}

// CHECK-LABEL: define void @_ZN8PR12768324test_volatile_qual_storeEv()
// CHECK:         store volatile i32 0, ptr @_ZN8PR1276833ObjE, align 4
void test_volatile_qual_store() {
  Obj.Base::Val = 0;
}

// CHECK-LABEL: define void @_ZN8PR12768323test_volatile_qual_loadEv()
// CHECK:         load volatile i32, ptr @_ZN8PR1276833ObjE, align 4
void test_volatile_qual_load() {
  [[maybe_unused]] int Val = Obj.Base::Val;
}

// CHECK-LABEL: define void @_ZN8PR12768328test_volatile_ptr_qual_storeEv()
// CHECK:         store volatile i32 0, ptr %{{.*}}, align 4
void test_volatile_ptr_qual_store() {
  Ptr->Base::Val = 0;
}

// CHECK-LABEL: define void @_ZN8PR12768327test_volatile_ptr_qual_loadEv()
// CHECK:         load volatile i32, ptr %{{.*}}, align 4
void test_volatile_ptr_qual_load() {
  [[maybe_unused]] int Val = Ptr->Base::Val;
}

struct DerivedMethod : Base {
  void test_this_volatile() volatile;
};

// CHECK-LABEL: define void @_ZNV8PR12768313DerivedMethod18test_this_volatileEv(
// CHECK:         store volatile i32 0, ptr %{{.*}}, align 4
// CHECK:         store volatile i32 1, ptr %{{.*}}, align 4
void DerivedMethod::test_this_volatile() volatile {
  Val = 0;
  Base::Val = 1;
}

// Ensure volatile on the pointer itself does not make the pointee access volatile.
// CHECK-LABEL: define void @_ZN8PR12768330test_volatile_ptr_itself_storeEv()
// CHECK:         load volatile ptr, ptr @_ZN8PR1276836VolPtrE, align 8
// CHECK:         store i32 0, ptr %{{.*}}, align 4
// CHECK:         load volatile ptr, ptr @_ZN8PR1276836VolPtrE, align 8
// CHECK:         store i32 1, ptr %{{.*}}, align 4
void test_volatile_ptr_itself_store() {
  VolPtr->Val = 0;
  VolPtr->Base::Val = 1;
}

// CHECK-LABEL: define void @_ZN8PR12768335test_volatile_ptr_and_pointee_storeEv()
// CHECK:         load volatile ptr, ptr @_ZN8PR1276837VPtrToVE, align 8
// CHECK:         store volatile i32 0, ptr %{{.*}}, align 4
void test_volatile_ptr_and_pointee_store() {
  VPtrToV->Val = 0;
}

struct Derived1 : Base {};
struct Derived2 : Base {};
struct VeryDerived : Derived1, Derived2 {};

// CHECK-LABEL: define void @_ZN8PR12768318test_diamond_storeERVNS_11VeryDerivedEPS1_(
// CHECK:         store volatile i32 0, ptr %{{.*}}, align 4
// CHECK:         store volatile i32 1, ptr %{{.*}}, align 4
void test_diamond_store(volatile VeryDerived &R, volatile VeryDerived *P) {
  R.Derived1::Val = 0;
  P->Derived1::Val = 1;
}

struct VDerived : virtual Base {};

// CHECK-LABEL: define void @_ZN8PR12768323test_virtual_base_storeERVNS_8VDerivedEPS1_(
// CHECK:         store volatile i32 0, ptr %{{.*}}, align 4
// CHECK:         store volatile i32 1, ptr %{{.*}}, align 4
// CHECK:         store volatile i32 2, ptr %{{.*}}, align 4
// CHECK:         store volatile i32 3, ptr %{{.*}}, align 4
void test_virtual_base_store(volatile VDerived &R, volatile VDerived *P) {
  R.Val = 0;
  R.Base::Val = 1;
  P->Val = 2;
  P->Base::Val = 3;
}

// CHECK-LABEL: define void @_ZN8PR12768327test_const_volatile_mutableERVKNS_7DerivedEPS1_(
// CHECK:         store volatile i32 0, ptr %{{.*}}, align 4
// CHECK:         store volatile i32 1, ptr %{{.*}}, align 4
// CHECK:         store volatile i32 2, ptr %{{.*}}, align 4
// CHECK:         store volatile i32 3, ptr %{{.*}}, align 4
void test_const_volatile_mutable(const volatile Derived &R,
                                 const volatile Derived *P) {
  R.MutVal = 0;
  R.Base::MutVal = 1;
  P->MutVal = 2;
  P->Base::MutVal = 3;
}

using VolBase = volatile Base;

// CHECK-LABEL: define void @_ZN8PR12768326test_qualified_typedef_nnsERNS_7DerivedEPS0_(
// CHECK:         store i32 0, ptr %{{.*}}, align 4
// CHECK:         store i32 1, ptr %{{.*}}, align 4
void test_qualified_typedef_nns(Derived &R, Derived *P) {
  R.VolBase::Val = 0;
  P->VolBase::Val = 1;
}

}

// CHECK: attributes [[NUW]] = { mustprogress noinline nounwind{{.*}} }
