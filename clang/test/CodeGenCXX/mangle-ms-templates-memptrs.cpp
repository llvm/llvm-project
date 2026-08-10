// RUN: %clang_cc1 -Wno-microsoft -fno-rtti -std=c++11 -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s

struct U;
static_assert(sizeof(void (U::*)()) == 2 * sizeof(void*) + 2 * sizeof(int), "");

struct A { int a; };
struct B { int b; };
struct I { union { struct { int a, b; }; }; };

struct S             { int a, b; void f(); virtual void g(); };
struct M : A, B      { int a, b; void f(); virtual void g(); };
struct V : virtual A { int a, b; void f(); virtual void g(); };
struct U             { int a, b; void f(); virtual void g(); };

struct C        { virtual void f(); };
struct D        { virtual void g(); };
struct O : C, D { virtual void g(); }; // override of non-primary

// Test data member pointers.
template <typename T, int T::*F>
int ReadField(T &o) {
  return F ? o.*F : 0;
}

// Redeclare some of the classes so that the implicit attribute goes on the most
// recent redeclaration rather than the definition.
struct V;

void ReadFields() {
  A a;
  I i;
  S s;
  M m;
  V v;
  U u;
  ReadField<S, &S::a>(s);
  ReadField<M, &M::a>(m);
  ReadField<V, &V::a>(v);
  ReadField<U, &U::a>(u);
  ReadField<S, &S::b>(s);
  ReadField<M, &M::b>(m);
  ReadField<V, &V::b>(v);
  ReadField<U, &U::b>(u);
  ReadField<S, nullptr>(s);
  ReadField<M, nullptr>(m);
  ReadField<V, nullptr>(v);
  ReadField<U, nullptr>(u);

  // Non-polymorphic null data memptr vs first field memptr.
  ReadField<A, &A::a>(a);
  ReadField<A, nullptr>(a);

  // Indirect fields injected from anonymous unions and structs
  ReadField<I, &I::a>(i);
  ReadField<I, &I::b>(i);
}

// CHECK-LABEL: define {{.*}}ReadFields
// CHECK: call {{.*}} @"??$ReadField@US@@$03@@YAHAAUS@@@Z"
// CHECK: call {{.*}} @"??$ReadField@UM@@$0M@@@YAHAAUM@@@Z"
// CHECK: call {{.*}} @"??$ReadField@UV@@$F7A@@@YAHAAUV@@@Z"
// CHECK: call {{.*}} @"??$ReadField@UU@@$G3A@A@@@YAHAAUU@@@Z"
// CHECK: call {{.*}} @"??$ReadField@US@@$07@@YAHAAUS@@@Z"
// CHECK: call {{.*}} @"??$ReadField@UM@@$0BA@@@YAHAAUM@@@Z"
// CHECK: call {{.*}} @"??$ReadField@UV@@$FM@A@@@YAHAAUV@@@Z"
// CHECK: call {{.*}} @"??$ReadField@UU@@$G7A@A@@@YAHAAUU@@@Z"

// MSVC mangles null member pointers in function templates wrong, but it gets
// them right in class templates.
// CHECK: call {{.*}} @"??$ReadField@US@@$0A@@@YAHAAUS@@@Z"
// CHECK: call {{.*}} @"??$ReadField@UM@@$0A@@@YAHAAUM@@@Z"
// CHECK: call {{.*}} @"??$ReadField@UV@@$0A@@@YAHAAUV@@@Z"
// CHECK: call {{.*}} @"??$ReadField@UU@@$0A@@@YAHAAUU@@@Z"

// Non-polymorphic null data memptr vs first field memptr.  MSVC mangles these
// the same.
// CHECK: call {{.*}} @"??$ReadField@UA@@$0A@@@YAHAAUA@@@Z"
// CHECK: call {{.*}} @"??$ReadField@UA@@$0?0@@YAHAAUA@@@Z"

// Indirect fields are handled as-if they were simply members of their enclosing
// record.
// CHECK: call {{.*}} @"??$ReadField@UI@@$0A@@@YAHAAUI@@@Z"
// CHECK: call {{.*}} @"??$ReadField@UI@@$03@@YAHAAUI@@@Z"

// Test member function pointers.
template <typename T, void (T::*MFP)()>
void CallMethod(T &o) {
  (o.*MFP)();
}

void CallMethods() {
  S s;
  M m;
  V v;
  U u;
  O o;

  // Non-virtual methods.
  CallMethod<S, &S::f>(s);
  CallMethod<M, &M::f>(m);
  CallMethod<V, &V::f>(v);
  CallMethod<U, &U::f>(u);

  // Virtual methods requiring thunk mangling.
  CallMethod<S, &S::g>(s);
  CallMethod<M, &M::g>(m);
  CallMethod<V, &V::g>(v);
  CallMethod<U, &U::g>(u);

  // A member pointer for a non-primary vbase will have a non-zero this
  // adjustment.
  CallMethod<O, &O::g>(o);

  // Null member pointers.
  CallMethod<S, nullptr>(s);
  CallMethod<M, nullptr>(m);
  CallMethod<V, nullptr>(v);
  CallMethod<U, nullptr>(u);
}

// CHECK-LABEL: define {{.*}}CallMethods
// CHECK: call {{.*}} @"??$CallMethod@US@@$1?f@1@QAEXXZ@@YAXAAUS@@@Z"
// CHECK: call {{.*}} @"??$CallMethod@UM@@$H?f@1@QAEXXZA@@@YAXAAUM@@@Z"
// CHECK: call {{.*}} @"??$CallMethod@UV@@$I?f@1@QAEXXZA@A@@@YAXAAUV@@@Z"
// CHECK: call {{.*}} @"??$CallMethod@UU@@$J?f@1@QAEXXZA@A@A@@@YAXAAUU@@@Z"

// PR17034: MSVC reuses the same thunk for every virtual g method because they
// are all at vftable offset zero.  They then mangle the name of the first thunk
// created into the name of the template instantiation, which is definitely a
// bug.  We don't follow them here.  Instead of ?_91@ backref below, they would
// get ?_9S@@ in every instantiation after the first.

// CHECK: call {{.*}} @"??$CallMethod@US@@$1??_91@$BA@AE@@YAXAAUS@@@Z"
// CHECK: call {{.*}} @"??$CallMethod@UM@@$H??_91@$BA@AEA@@@YAXAAUM@@@Z"
// CHECK: call {{.*}} @"??$CallMethod@UV@@$I??_91@$BA@AEA@A@@@YAXAAUV@@@Z"
// CHECK: call {{.*}} @"??$CallMethod@UU@@$J??_91@$BA@AEA@A@A@@@YAXAAUU@@@Z"

// CHECK: call {{.*}} @"??$CallMethod@UO@@$H??_91@$BA@AE3@@YAXAAUO@@@Z"

// CHECK: call {{.*}} @"??$CallMethod@US@@$0A@@@YAXAAUS@@@Z"
// CHECK: call {{.*}} @"??$CallMethod@UM@@$0A@@@YAXAAUM@@@Z"
// CHECK: call {{.*}} @"??$CallMethod@UV@@$0A@@@YAXAAUV@@@Z"
// CHECK: call {{.*}} @"??$CallMethod@UU@@$0A@@@YAXAAUU@@@Z"

namespace NegativeNVOffset {
struct A {};
struct B : virtual A {};
struct C : B {
  virtual void f();
};
}

template void CallMethod<NegativeNVOffset::C, &NegativeNVOffset::C::f>(NegativeNVOffset::C &);

// CHECK-LABEL: define {{.*}} @"??$CallMethod@UC@NegativeNVOffset@@$I??_912@$BA@AEPPPPPPPM@A@@@YAXAAUC@NegativeNVOffset@@@Z"
