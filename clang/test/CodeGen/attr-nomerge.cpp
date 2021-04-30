// RUN: %clang_cc1 -S -emit-llvm %s -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

class A {
public:
  [[clang::nomerge]] A();
  [[clang::nomerge]] virtual ~A();
  [[clang::nomerge]] void f();
  [[clang::nomerge]] virtual void g();
  [[clang::nomerge]] static void f1();
};

class B : public A {
public:
  void g() override;
};

bool bar();
[[clang::nomerge]] void f(bool, bool);

void foo(int i, A *ap, B *bp) {
  [[clang::nomerge]] bar();
  [[clang::nomerge]] (i = 4, bar());
  [[clang::nomerge]] (void)(bar());
  f(bar(), bar());
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

  A *newA = new B();
  delete newA;
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

// CHECK: call zeroext i1 @_Z3barv() #[[ATTR0:[0-9]+]]
// CHECK: call zeroext i1 @_Z3barv() #[[ATTR0]]
// CHECK: call zeroext i1 @_Z3barv() #[[ATTR0]]
// CHECK: call zeroext i1 @_Z3barv(){{$}}
// CHECK: call zeroext i1 @_Z3barv(){{$}}
// CHECK: call void @_Z1fbb({{.*}}) #[[ATTR0]]
// CHECK: call void @"_ZZ3fooiP1AP1BENK3$_0clEv"{{.*}} #[[ATTR0]]
// CHECK: call zeroext i1 @_Z3barv() #[[ATTR0]]
// CHECK-LABEL: for.cond:
// CHECK: call zeroext i1 @_Z3barv() #[[ATTR0]]
// CHECK-LABEL: for.inc:
// CHECK: call zeroext i1 @_Z3barv() #[[ATTR0]]
// CHECK: call void asm sideeffect "nop"{{.*}} #[[ATTR1:[0-9]+]]
// CHECK: call zeroext i1 @_Z3barv(){{$}}
// CHECK: %[[AG:.*]] = load void (%class.A*)*, void (%class.A*)**
// CHECK-NEXT: call void %[[AG]](%class.A* {{.*}}) #[[ATTR0]]
// CHECK: %[[BG:.*]] = load void (%class.B*)*, void (%class.B*)**
// CHECK-NEXT: call void %[[BG]](%class.B* nonnull dereferenceable
// CHECK: call void @_ZN1AC1Ev({{.*}}) #[[ATTR0]]
// CHECK: call void @_ZN1A1fEv({{.*}}) #[[ATTR0]]
// CHECK: call void @_ZN1A1gEv({{.*}}) #[[ATTR0]]
// CHECK: call void @_ZN1A2f1Ev() #[[ATTR0]]
// CHECK: call void @_ZN1BC1Ev({{.*}}){{$}}
// CHECK: call void @_ZN1B1gEv({{.*}}){{$}}
// CHECK: call void @_ZN1BC1Ev({{.*}}){{$}}
// CHECK: %[[AG:.*]] = load void (%class.A*)*, void (%class.A*)**
// CHECK-NEXT: call void %[[AG]](%class.A* {{.*}}) #[[ATTR1]]
// CHECK: call void  @_ZN1AD1Ev(%class.A* {{.*}}) #[[ATTR1]]

// CHECK-DAG: attributes #[[ATTR0]] = {{{.*}}nomerge{{.*}}}
// CHECK-DAG: attributes #[[ATTR1]] = {{{.*}}nomerge{{.*}}}
