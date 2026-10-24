// Check that clang_getCursorReferenced() resolves member references through a
// dependent base class to the member of the primary template.

template <typename T>
class A {
public:
  int size() const { return 1; }
  int field;
  void ov(int);
  void ov(float);
};

template <typename T>
class B : public A<T> {
public:
  int f1() const { return this->size(); }
  int f2() const { return this->field; }
  void f3() { this->ov(1); }
  int f4() const { return A<T>::size(); }
};

// RUN: c-index-test -test-load-source local %s | FileCheck %s

// CHECK: 16:27: CallExpr=:7:7 Extent=[16:27 - 16:39]
// CHECK: 16:27: MemberRefExpr=:7:7 Extent=[16:27 - 16:37]
// CHECK: 17:27: MemberRefExpr=:8:7 Extent=[17:27 - 17:38]

// An overloaded member can't be resolved to a single declaration.
// CHECK: 18:15: MemberRefExpr= Extent=[18:15 - 18:23]

// HeuristicResolver only considers static members for a qualified dependent
// name, so 'A<T>::size()' isn't resolved yet.
// CHECK: 19:27: DeclRefExpr= Extent=[19:27 - 19:37]

// Only dependent member names are resolved. A call's callee is looked through
// as for non-dependent calls, so 'm.make()()' refers to 'make', not to the
// class it returns. A dependent call found by ADL isn't resolved, since other
// candidates may be found at instantiation.
template <typename T> struct Functor { void operator()(); };
template <typename T> struct Maker { Functor<T> make(); };
void g(int);
template <typename T> void h1(Maker<T> m) { m.make()(); }
template <typename T> void h2(T t) { g(t); }

// CHECK: 42:45: CallExpr=:40:49 Extent=[42:45 - 42:55]
// CHECK: 42:45: CallExpr=:40:49 Extent=[42:45 - 42:53]
// CHECK: 42:45: MemberRefExpr=:40:49 Extent=[42:45 - 42:51]
// CHECK: 43:38: CallExpr= Extent=[43:38 - 43:42]
