// RUN: %clang_cc1 -std=c++11 -fexceptions -fcxx-exceptions -fsyntax-only -verify %s
// RUN: %clang_cc1 -DUSE -std=c++11 -fexceptions -fcxx-exceptions -fsyntax-only -verify %s

// Maybe force the implicit declaration of 'operator delete' and 'operator
// delete[]'. This should make no difference to anything!
#ifdef USE
void f(int *p) {
  delete p;
  delete [] p;
}
#endif

// Deallocation functions are implicitly noexcept.
// Thus, explicit specs aren't allowed to conflict.

void operator delete(void*); // expected-warning {{function previously declared with an explicit exception specification redeclared with an implicit exception specification}}
void operator delete[](void*); // expected-warning {{function previously declared with an explicit exception specification redeclared with an implicit exception specification}}

static_assert(noexcept(operator delete(0)), "");
static_assert(noexcept(operator delete[](0)), "");

// Same goes for explicit declarations.
void operator delete(void*, float);
void operator delete[](void*, float);

static_assert(noexcept(operator delete(0, 0.f)), "");
static_assert(noexcept(operator delete[](0, 0.f)), "");

// But explicit specs stay.
void operator delete(void*, double) throw(int); // expected-note {{previous}}
static_assert(!noexcept(operator delete(0, 0.)), "");
void operator delete(void*, double) noexcept; // expected-error {{does not match}}
