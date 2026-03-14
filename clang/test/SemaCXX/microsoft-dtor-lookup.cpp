// RUN: %clang_cc1 -fexceptions -fcxx-exceptions -std=c++11 -triple %itanium_abi_triple -fsyntax-only %s
// RUN: %clang_cc1 -fexceptions -fcxx-exceptions -std=c++11 -triple %ms_abi_triple -verify %s

namespace Test1 {

// Should be accepted under the Itanium ABI (first RUN line) but rejected
// under the Microsoft ABI (second RUN line), as Microsoft ABI requires
// operator delete() lookups to be done when vtables are marked used.

struct A {
  void operator delete(void *); // expected-note {{member found by ambiguous name lookup}}
};

struct B {
  void operator delete(void *); // expected-note {{member found by ambiguous name lookup}}
};

struct C : A, B {
  ~C();
};

struct VC : A, B {
  virtual ~VC(); // expected-error {{member 'operator delete' found in multiple base classes of different types}}
};

void f() {
  // This marks VC's vtable used.
  VC vc;
}

}

namespace Test2 {

// In the MSVC ABI, functions must destroy their aggregate arguments.  foo
// requires a dtor for B, but we can't implicitly define it because ~A is
// private.  bar should be able to call A's private dtor without error, even
// though MSVC rejects bar.
class A {
private:
  ~A();
  int a;
};

struct B : public A { // expected-note {{destructor of 'B' is implicitly deleted because base class 'A' has an inaccessible destructor}}
  int b;
};

struct C {
  ~C();
  int c;
};

struct D {
  // D has a non-trivial implicit dtor that destroys C.
  C o;
};

void foo(B b) { } // expected-error {{attempt to use a deleted function}}
void bar(A a) { } // no error; MSVC rejects this, but we skip the direct access check.
void baz(D d) { } // no error

}

#ifdef MSVC_ABI
namespace Test3 {

class A {
  A();
  ~A(); // expected-note {{implicitly declared private here}}
  friend void bar(A);
  int a;
};

void bar(A a) { }
void baz(A a) { } // no error; MSVC rejects this, but the standard allows it.

// MSVC accepts foo() but we reject it for consistency with Itanium.  MSVC also
// rejects this if A has a copy ctor or if we call A's ctor.
void foo(A *a) {
  bar(*a); // expected-error {{temporary of type 'Test3::A' has private destructor}}
}
}
#endif

namespace Test4 {
// Don't try to access the dtor of an incomplete on a function declaration.
class A;
void foo(A a);
}

#ifdef MSVC_ABI
namespace Test5 {
// Do the operator delete access control check from the context of the dtor.
class A {
 protected:
  void operator delete(void *);
};
class B : public A {
  virtual ~B();
};
B *test() {
  // Previously, marking the vtable used here would do the operator delete
  // lookup from this context, which doesn't have access.
  return new B;
}
}
#endif

namespace Test6 {
class A {
protected:
  void operator delete(void *);
};
class B : public A {
  virtual ~B();
public:
  virtual void m_fn1();
};
void fn1(B *b) { b->m_fn1(); }
}

namespace Test7 {
class A {
protected:
  void operator delete(void *);
};
struct B : public A {
  virtual ~B();
};
void fn1(B b) {}
}
