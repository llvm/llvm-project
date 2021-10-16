// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -o %t.ll -O1 -disable-llvm-passes -fms-extensions -fstrict-vtable-pointers
// RUN: %clang_cc1 %s -triple i686-pc-win32 -emit-llvm -o %t.ms.ll -O1 -disable-llvm-passes -fms-extensions -fstrict-vtable-pointers
// FIXME: Assume load should not require -fstrict-vtable-pointers

// RUN: FileCheck --check-prefix=CHECK1 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK2 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK3 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK4 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK-MS --input-file=%t.ms.ll %s
// RUN: FileCheck --check-prefix=CHECK6 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK7 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK8 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK9 --input-file=%t.ll %s
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
// CHECK1: call void @_ZN5test11AC1Ev(%"struct.test1::A"*
// CHECK1: %[[VTABLE:.*]] = load i8**, i8*** %{{.*}}
// CHECK1: %[[CMP:.*]] = icmp eq i8** %[[VTABLE]], getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTVN5test11AE, i32 0, inrange i32 0, i32 2)
// CHECK1: call void @llvm.assume(i1 %[[CMP]])
// CHECK1-LABEL: {{^}}}

void fooA() {
  A a;
  g(&a);
}

// CHECK1-LABEL: define{{.*}} void @_ZN5test14fooBEv()
// CHECK1: call void @_ZN5test11BC1Ev(%"struct.test1::B"* {{[^,]*}} %{{.*}})
// CHECK1: %[[VTABLE:.*]] = load i8**, i8*** %{{.*}}
// CHECK1: %[[CMP:.*]] = icmp eq i8** %[[VTABLE]], getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTVN5test11BE, i32 0, inrange i32 0, i32 2)
// CHECK1: call void @llvm.assume(i1 %[[CMP]])
// CHECK1-LABEL: {{^}}}

void fooB() {
  B b;
  g(&b);
}
// there should not be any assumes in the ctor that calls base ctor
// CHECK1-LABEL: define linkonce_odr void @_ZN5test11BC2Ev(%"struct.test1::B"*
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
// CHECK2: call void @_ZN5test21CC1Ev(%"struct.test2::C"*
// CHECK2: %[[VTABLE:.*]] = load i8**, i8*** {{.*}}
// CHECK2: %[[CMP:.*]] = icmp eq i8** %[[VTABLE]], getelementptr inbounds ({ [3 x i8*], [3 x i8*] }, { [3 x i8*], [3 x i8*] }* @_ZTVN5test21CE, i32 0, inrange i32 0, i32 2)
// CHECK2: call void @llvm.assume(i1 %[[CMP]])

// CHECK2: %[[V2:.*]] = bitcast %"struct.test2::C"* %{{.*}} to i8*
// CHECK2: %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* %[[V2]], i64 8
// CHECK2: %[[V3:.*]] = bitcast i8* %[[ADD_PTR]] to i8***
// CHECK2: %[[VTABLE2:.*]] = load i8**, i8*** %[[V3]]
// CHECK2: %[[CMP2:.*]] = icmp eq i8** %[[VTABLE2]], getelementptr inbounds ({ [3 x i8*], [3 x i8*] }, { [3 x i8*], [3 x i8*] }* @_ZTVN5test21CE, i32 0, inrange i32 1, i32 2)
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
// CHECK3: call void @_ZN5test31CC1Ev(%"struct.test3::C"*
// CHECK3: %[[CMP:.*]] = icmp eq i8** %{{.*}}, getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN5test31CE, i32 0, inrange i32 0, i32 3)
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
// CHECK4: call void @_ZN5test41CC1Ev(%"struct.test4::C"*
// CHECK4: %[[VTABLE:.*]] = load i8**, i8*** %{{.*}}
// CHECK4: %[[CMP:.*]] = icmp eq i8** %[[VTABLE]], getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTVN5test41CE, i32 0, inrange i32 0, i32 4)
// CHECK4: call void @llvm.assume(i1 %[[CMP]]

// CHECK4: %[[VTABLE2:.*]] = load i8**, i8*** %{{.*}}
// CHECK4: %[[CMP2:.*]] = icmp eq i8** %[[VTABLE2]], getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTVN5test41CE, i32 0, inrange i32 0, i32 4)
// CHECK4: call void @llvm.assume(i1 %[[CMP2]])
// CHECK4-LABEL: {{^}}}

void test() {
  C c;
  g(&c);
}
} // test4

namespace testMS {

struct __declspec(novtable) S {
  virtual void foo();
};

void g(S &s) { s.foo(); }

// if struct has novtable specifier, then we can't generate assumes
// CHECK-MS-LABEL: define dso_local void @"?test@testMS@@YAXXZ"()
// CHECK-MS: call x86_thiscallcc noundef %"struct.testMS::S"* @"??0S@testMS@@QAE@XZ"(
// CHECK-MS-NOT: @llvm.assume
// CHECK-MS-LABEL: {{^}}}

void test() {
  S s;
  g(s);
}

} // testMS

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
// CHECK6-LABEL: define{{.*}} void @_ZN5test61gEv()
// CHECK6: call void @_ZN5test61AC1Ev(
// CHECK6-NOT: call void @llvm.assume(

// We can't emit assumption loads for B, because if we would refer to vtable
// it would refer to functions that will not be able to find (like implicit
// inline destructor).

// CHECK6-LABEL:   call void @_ZN5test61BC1Ev(
// CHECK6-NOT: call void @llvm.assume(
// CHECK6-LABEL: {{^}}}
void g() {
  A *a = new A;
  B *b = new B;
}
}

namespace test7 {
// Because A's key function is defined here, vtable is generated in this TU
// CHECK7: @_ZTVN5test71AE ={{.*}} unnamed_addr constant
struct A {
  A();
  virtual void foo();
  virtual void bar();
};
void A::foo() {}

// CHECK7-LABEL: define{{.*}} void @_ZN5test71gEv()
// CHECK7: call void @_ZN5test71AC1Ev(
// CHECK7: call void @llvm.assume(
// CHECK7-LABEL: {{^}}}
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

// CHECK8-DAG: @_ZTVN5test81BE = available_externally unnamed_addr constant
struct B : A {
  B();
  void foo();
  void bar();
};

// CHECK8-DAG: @_ZTVN5test81CE = linkonce_odr unnamed_addr constant
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

// CHECK8-DAG: @_ZTVN5test81EE = linkonce_odr unnamed_addr constant
struct E : A {
  E();
};

// CHECK8-LABEL: define{{.*}} void @_ZN5test81bEv()
// CHECK8: call void @llvm.assume(
// CHECK8-LABEL: {{^}}}
void b() {
  B b;
  b.bar();
}

// FIXME: C has inline virtual functions which prohibits as from generating
// assumption loads, but because vtable is generated in this TU (key function
// defined here) it would be correct to refer to it.
// CHECK8-LABEL: define{{.*}} void @_ZN5test81cEv()
// CHECK8-NOT: call void @llvm.assume(
// CHECK8-LABEL: {{^}}}
void c() {
  C c;
  c.bar();
}

// FIXME: We could generate assumption loads here.
// CHECK8-LABEL: define{{.*}} void @_ZN5test81dEv()
// CHECK8-NOT: call void @llvm.assume(
// CHECK8-LABEL: {{^}}}
void d() {
  D d;
  d.bar();
}

// CHECK8-LABEL: define{{.*}} void @_ZN5test81eEv()
// CHECK8: call void @llvm.assume(
// CHECK8-LABEL: {{^}}}
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

// CHECK9-LABEL: define{{.*}} void @_ZN5test94testEv()
// CHECK9-NOT: @llvm.assume(
// CHECK9: }
void test() {
  S *s = new S();
  s->doStuff();
  delete s;
}
}

