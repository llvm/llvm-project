// RUN: %clang_cc1 -emit-llvm %s -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

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
[[clang::nomerge]] void (*fptr)(void);

void foo(int i, A *ap, B *bp) {
  [[clang::nomerge]] bar();
  [[clang::nomerge]] (i = 4, bar());
  [[clang::nomerge]] (void)(bar());
  f(bar(), bar());
  fptr();
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

// CHECK: call noundef zeroext i1 @_Z3barv() #[[ATTR0:[0-9]+]]
// CHECK: call noundef zeroext i1 @_Z3barv() #[[ATTR0]]
// CHECK: call noundef zeroext i1 @_Z3barv() #[[ATTR0]]
// CHECK: call noundef zeroext i1 @_Z3barv(){{$}}
// CHECK: call noundef zeroext i1 @_Z3barv(){{$}}
// CHECK: call void @_Z1fbb({{.*}}) #[[ATTR0]]
// CHECK: %[[FPTR:.*]] = load ptr, ptr @fptr
// CHECK-NEXT: call void %[[FPTR]]() #[[ATTR0]]
// CHECK: call void @"_ZZ3fooiP1AP1BENK3$_0clEv"{{.*}} #[[ATTR0]]
// CHECK: call noundef zeroext i1 @_Z3barv() #[[ATTR0]]
// CHECK-LABEL: for.cond:
// CHECK: call noundef zeroext i1 @_Z3barv() #[[ATTR0]]
// CHECK-LABEL: for.inc:
// CHECK: call noundef zeroext i1 @_Z3barv() #[[ATTR0]]
// CHECK: call void asm sideeffect "nop"{{.*}} #[[ATTR1:[0-9]+]]
// CHECK: call noundef zeroext i1 @_Z3barv(){{$}}
// CHECK: load ptr, ptr
// CHECK: load ptr, ptr
// CHECK: %[[AG:.*]] = load ptr, ptr
// CHECK-NEXT: call void %[[AG]](ptr {{.*}}) #[[ATTR0]]
// CHECK: load ptr, ptr
// CHECK: load ptr, ptr
// CHECK: %[[BG:.*]] = load ptr, ptr
// CHECK-NEXT: call void %[[BG]](ptr noundef{{.*}}
// CHECK: call void @_ZN1AC1Ev({{.*}}) #[[ATTR0]]
// CHECK: call void @_ZN1A1fEv({{.*}}) #[[ATTR0]]
// CHECK: call void @_ZN1A1gEv({{.*}}) #[[ATTR0]]
// CHECK: call void @_ZN1A2f1Ev() #[[ATTR0]]
// CHECK: call void @_ZN1BC1Ev({{.*}}){{$}}
// CHECK: call void @_ZN1B1gEv({{.*}}){{$}}
// CHECK: call void @_ZN1BC1Ev({{.*}}){{$}}
// CHECK: load ptr, ptr
// CHECK: load ptr, ptr
// CHECK: %[[AG:.*]] = load ptr, ptr
// CHECK-NEXT: call void %[[AG]](ptr {{.*}}) #[[ATTR1]]
// CHECK: call void  @_ZN1AD1Ev(ptr {{.*}}) #[[ATTR1]]

// CHECK-DAG: attributes #[[ATTR0]] = {{{.*}}nomerge{{.*}}}
// CHECK-DAG: attributes #[[ATTR1]] = {{{.*}}nomerge{{.*}}}
