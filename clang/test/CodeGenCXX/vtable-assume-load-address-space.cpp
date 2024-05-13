// RUN: %clang_cc1 %s -triple=amdgcn-amd-amdhsa -std=c++11 -emit-llvm -o %t.ll -O1 -disable-llvm-passes -fms-extensions -fstrict-vtable-pointers
// FIXME: Assume load should not require -fstrict-vtable-pointers

// RUN: FileCheck --check-prefix=CHECK1 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK2 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK3 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK4 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK5 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK6 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK7 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK8 --input-file=%t.ll %s
namespace test1 {

struct A {
  A();
  virtual void foo();
};

struct B : A {
  virtual void foo();
};

void g(A *a) { a->foo(); }

// CHECK1-LABEL: define{{.*}} void @_ZN5test14fooAEv()
// CHECK1: call void @_ZN5test11AC1Ev(ptr
// CHECK1: %[[VTABLE:.*]] = load ptr addrspace(1), ptr %{{.*}}
// CHECK1: %[[CMP:.*]] = icmp eq ptr addrspace(1) %[[VTABLE]], getelementptr inbounds inrange(-16, 8) ({ [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5test11AE, i32 0, i32 0, i32 2)
// CHECK1: call void @llvm.assume(i1 %[[CMP]])
// CHECK1-LABEL: {{^}}}

void fooA() {
  A a;
  g(&a);
}

// CHECK1-LABEL: define{{.*}} void @_ZN5test14fooBEv()
// CHECK1: call void @_ZN5test11BC1Ev(ptr {{[^,]*}} %{{.*}})
// CHECK1: %[[VTABLE:.*]] = load ptr addrspace(1), ptr %{{.*}}
// CHECK1: %[[CMP:.*]] = icmp eq ptr addrspace(1) %[[VTABLE]], getelementptr inbounds inrange(-16, 8) ({ [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5test11BE, i32 0, i32 0, i32 2)
// CHECK1: call void @llvm.assume(i1 %[[CMP]])
// CHECK1-LABEL: {{^}}}

void fooB() {
  B b;
  g(&b);
}
// there should not be any assumes in the ctor that calls base ctor
// CHECK1-LABEL: define linkonce_odr void @_ZN5test11BC2Ev(ptr
// CHECK1-NOT: @llvm.assume(
// CHECK1-LABEL: {{^}}}
}
namespace test2 {
struct A {
  A();
  virtual void foo();
};

struct B {
  B();
  virtual void bar();
};

struct C : A, B {
  C();
  virtual void foo();
};
void g(A *a) { a->foo(); }
void h(B *b) { b->bar(); }

// CHECK2-LABEL: define{{.*}} void @_ZN5test24testEv()
// CHECK2: call void @_ZN5test21CC1Ev(ptr
// CHECK2: %[[VTABLE:.*]] = load ptr addrspace(1), ptr {{.*}}
// CHECK2: %[[CMP:.*]] = icmp eq ptr addrspace(1) %[[VTABLE]], getelementptr inbounds inrange(-16, 8) ({ [3 x ptr addrspace(1)], [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5test21CE, i32 0, i32 0, i32 2)
// CHECK2: call void @llvm.assume(i1 %[[CMP]])

// CHECK2: %[[ADD_PTR:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i64 8
// CHECK2: %[[VTABLE2:.*]] = load ptr addrspace(1), ptr %[[ADD_PTR]]
// CHECK2: %[[CMP2:.*]] = icmp eq ptr addrspace(1) %[[VTABLE2]], getelementptr inbounds inrange(-16, 8) ({ [3 x ptr addrspace(1)], [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5test21CE, i32 0, i32 1, i32 2)
// CHECK2: call void @llvm.assume(i1 %[[CMP2]])

// CHECK2: call void @_ZN5test21gEPNS_1AE(
// CHECK2-LABEL: {{^}}}

void test() {
  C c;
  g(&c);
  h(&c);
}
}

namespace test3 {
struct A {
  A();
};

struct B : A {
  B();
  virtual void foo();
};

struct C : virtual A, B {
  C();
  virtual void foo();
};
void g(B *a) { a->foo(); }

// CHECK3-LABEL: define{{.*}} void @_ZN5test34testEv()
// CHECK3: call void @_ZN5test31CC1Ev(ptr
// CHECK3: %[[CMP:.*]] = icmp eq ptr addrspace(1) %{{.*}}, getelementptr inbounds inrange(-24, 8) ({ [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5test31CE, i32 0, i32 0, i32 3)
// CHECK3: call void @llvm.assume(i1 %[[CMP]])
// CHECK3-LABLEL: }
void test() {
  C c;
  g(&c);
}
} // test3

namespace test4 {
struct A {
  A();
  virtual void foo();
};

struct B : virtual A {
  B();
  virtual void foo();
};
struct C : B {
  C();
  virtual void foo();
};

void g(C *c) { c->foo(); }

// CHECK4-LABEL: define{{.*}} void @_ZN5test44testEv()
// CHECK4: call void @_ZN5test41CC1Ev(ptr
// CHECK4: %[[VTABLE:.*]] = load ptr addrspace(1), ptr %{{.*}}
// CHECK4: %[[CMP:.*]] = icmp eq ptr addrspace(1) %[[VTABLE]], getelementptr inbounds inrange(-32, 8) ({ [5 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5test41CE, i32 0, i32 0, i32 4)
// CHECK4: call void @llvm.assume(i1 %[[CMP]]

// CHECK4: %[[VTABLE2:.*]] = load ptr addrspace(1), ptr %{{.*}}
// CHECK4: %[[CMP2:.*]] = icmp eq ptr addrspace(1) %[[VTABLE2]], getelementptr inbounds inrange(-32, 8) ({ [5 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5test41CE, i32 0, i32 0, i32 4)
// CHECK4: call void @llvm.assume(i1 %[[CMP2]])
// CHECK4-LABEL: {{^}}}

void test() {
  C c;
  g(&c);
}
} // test4

namespace test6 {
struct A {
  A();
  virtual void foo();
  virtual ~A() {}
};
struct B : A {
  B();
};
// FIXME: Because A's vtable is external, and no virtual functions are hidden,
// it's safe to generate assumption loads.
// CHECK5-LABEL: define{{.*}} void @_ZN5test61gEv()
// CHECK5: call void @_ZN5test61AC1Ev(
// CHECK5-NOT: call void @llvm.assume(

// We can't emit assumption loads for B, because if we would refer to vtable
// it would refer to functions that will not be able to find (like implicit
// inline destructor).

// CHECK5-LABEL:   call void @_ZN5test61BC1Ev(
// CHECK5-NOT: call void @llvm.assume(
// CHECK5-LABEL: {{^}}}
void g() {
  A *a = new A;
  B *b = new B;
}
}

namespace test7 {
// Because A's key function is defined here, vtable is generated in this TU
// CHECK6: @_ZTVN5test71AE ={{.*}} unnamed_addr addrspace(1) constant
struct A {
  A();
  virtual void foo();
  virtual void bar();
};
void A::foo() {}

// CHECK6-LABEL: define{{.*}} void @_ZN5test71gEv()
// CHECK6: call void @_ZN5test71AC1Ev(
// CHECK6: call void @llvm.assume(
// CHECK6-LABEL: {{^}}}
void g() {
  A *a = new A();
  a->bar();
}
}

namespace test8 {

struct A {
  virtual void foo();
  virtual void bar();
};

// CHECK7-DAG: @_ZTVN5test81BE = available_externally unnamed_addr addrspace(1) constant
struct B : A {
  B();
  void foo();
  void bar();
};

// CHECK7-DAG: @_ZTVN5test81CE = linkonce_odr unnamed_addr addrspace(1) constant
struct C : A {
  C();
  void bar();
  void foo() {}
};
inline void C::bar() {}

struct D : A {
  D();
  void foo();
  void inline bar();
};
void D::bar() {}

// CHECK7-DAG: @_ZTVN5test81EE = linkonce_odr unnamed_addr addrspace(1) constant
struct E : A {
  E();
};

// CHECK7-LABEL: define{{.*}} void @_ZN5test81bEv()
// CHECK7: call void @llvm.assume(
// CHECK7-LABEL: {{^}}}
void b() {
  B b;
  b.bar();
}

// FIXME: C has inline virtual functions which prohibits as from generating
// assumption loads, but because vtable is generated in this TU (key function
// defined here) it would be correct to refer to it.
// CHECK7-LABEL: define{{.*}} void @_ZN5test81cEv()
// CHECK7-NOT: call void @llvm.assume(
// CHECK7-LABEL: {{^}}}
void c() {
  C c;
  c.bar();
}

// FIXME: We could generate assumption loads here.
// CHECK7-LABEL: define{{.*}} void @_ZN5test81dEv()
// CHECK7-NOT: call void @llvm.assume(
// CHECK7-LABEL: {{^}}}
void d() {
  D d;
  d.bar();
}

// CHECK7-LABEL: define{{.*}} void @_ZN5test81eEv()
// CHECK7: call void @llvm.assume(
// CHECK7-LABEL: {{^}}}
void e() {
  E e;
  e.bar();
}
}

namespace test9 {

struct S {
  S();
  __attribute__((visibility("hidden"))) virtual void doStuff();
};

// CHECK8-LABEL: define{{.*}} void @_ZN5test94testEv()
// CHECK8-NOT: @llvm.assume(
// CHECK8: }
void test() {
  S *s = new S();
  s->doStuff();
  delete s;
}
}

