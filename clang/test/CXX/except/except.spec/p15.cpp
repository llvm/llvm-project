// RUN: %clang_cc1 -std=c++11 -fexceptions -fcxx-exceptions -fsyntax-only -verify %s

// Deallocation functions are implicitly noexcept.
// Thus, explicit specs aren't allowed to conflict.

void f() {
  // Force implicit declaration of delete.
  delete new int;
  delete[] new int[1];
}

void operator delete(void*);
void operator delete[](void*);

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
