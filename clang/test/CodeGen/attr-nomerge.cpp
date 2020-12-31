// RUN: %clang_cc1 -S -emit-llvm %s -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

class A {
public:
  [[clang::nomerge]] A();
  [[clang::nomerge]] ~A();
  [[clang::nomerge]] void f();
  [[clang::nomerge]] virtual void g();
  [[clang::nomerge]] static void f1();
};

class B : public A {
public:
  void g() override;
};

[[clang::nomerge]] bool bar();
[[clang::nomerge]] void f(bool, bool);

void foo(int i, A *ap, B *bp) {
  [[clang::nomerge]] bar();
  [[clang::nomerge]] (i = 4, bar());
  [[clang::nomerge]] (void)(bar());
  [[clang::nomerge]] f(bar(), bar());
  [[clang::nomerge]] [] { bar(); bar(); }(); // nomerge only applies to the anonymous function call
  [[clang::nomerge]] for (bar(); bar(); bar()) {}
  [[clang::nomerge]] { asm("nop"); }
  bar();

  ap->g();
  bp->g();

  A a;
  a.f();
  a.g();
  A::f1();

  B b;
  b.g();
}

int g(int i);

void something() {
  g(1);
}

[[clang::nomerge]] int g(int i);

void something_else() {
  g(1);
}

int g(int i) { return i; }

void something_else_again() {
  g(1);
}

// CHECK: call zeroext i1 @_Z3barv(){{$}}
// CHECK: call zeroext i1 @_Z3barv(){{$}}
// CHECK: call zeroext i1 @_Z3barv(){{$}}
// CHECK: call zeroext i1 @_Z3barv(){{$}}
// CHECK: call zeroext i1 @_Z3barv(){{$}}
// CHECK: call void @_Z1fbb({{.*}}){{$}}
// CHECK: call void @"_ZZ3fooiP1AP1BENK3$_0clEv"{{.*}} #[[ATTR0:[0-9]+]]
// CHECK: call zeroext i1 @_Z3barv(){{$}}
// CHECK: call zeroext i1 @_Z3barv(){{$}}
// CHECK: call zeroext i1 @_Z3barv(){{$}}
// CHECK: call void asm sideeffect "nop"{{.*}} #[[ATTR1:[0-9]+]]
// CHECK: call zeroext i1 @_Z3barv(){{$}}
// CHECK: %[[AG:.*]] = load void (%class.A*)*, void (%class.A*)**
// CHECK-NEXT: call void %[[AG]](%class.A* nonnull dereferenceable
// CHECK: %[[BG:.*]] = load void (%class.B*)*, void (%class.B*)**
// CHECK-NEXT: call void %[[BG]](%class.B* nonnull dereferenceable


// CHECK-DAG: declare zeroext i1 @_Z3barv() #[[ATTR2:[0-9]+]]
// CHECK-DAG: declare void @_Z1fbb(i1 zeroext, i1 zeroext) #[[ATTR2]]
// CHECK-DAG: declare void @_ZN1AC1Ev{{.*}} #[[ATTR2]]
// CHECK-DAG: declare void @_ZN1A1fEv{{.*}} #[[ATTR2]]
// CHECK-DAG: declare void @_ZN1A1gEv{{.*}} #[[ATTR2]]
// CHECK-DAG: declare void @_ZN1A2f1Ev{{.*}} #[[ATTR2]]
// CHECK-DAG: declare void @_ZN1AC2Ev{{.*}} #[[ATTR2]]
// CHECK-DAG: declare void @_ZN1AD1Ev{{.*}} #[[ATTR3:[0-9]+]]
// CHECK-DAG: declare void @_ZN1AD2Ev{{.*}} #[[ATTR3]]
// CHECK-DAG: define{{.*}} i32 @_Z1gi(i32 %i) #[[ATTR4:[0-9]+]] {

// CHECK-DAG: attributes #[[ATTR0]] = {{{.*}}nomerge{{.*}}}
// CHECK-DAG: attributes #[[ATTR1]] = {{{.*}}nomerge{{.*}}}
// CHECK-DAG: attributes #[[ATTR2]] = {{{.*}}nomerge{{.*}}}
// CHECK-DAG: attributes #[[ATTR3]] = {{{.*}}nomerge{{.*}}}
// CHECK-DAG: attributes #[[ATTR4]] = {{{.*}}nomerge{{.*}}}
