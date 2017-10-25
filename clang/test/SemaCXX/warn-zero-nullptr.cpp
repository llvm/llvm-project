// RUN: %clang_cc1 -fsyntax-only -verify %s -Wzero-as-null-pointer-constant -std=c++11

struct S {};

int (S::*mp0) = nullptr;
void (*fp0)() = nullptr;
void* p0 = nullptr;

int (S::*mp1) = 0; // expected-warning{{zero as null pointer constant}}
void (*fp1)() = 0; // expected-warning{{zero as null pointer constant}}
void* p1 = 0; // expected-warning{{zero as null pointer constant}}

// NULL is an integer constant expression, so warn on it too:
void* p2 = __null; // expected-warning{{zero as null pointer constant}}
void (*fp2)() = __null; // expected-warning{{zero as null pointer constant}}
int (S::*mp2) = __null; // expected-warning{{zero as null pointer constant}}

void f0(void* v = 0); // expected-warning{{zero as null pointer constant}}
void f1(void* v);

void g() {
  f1(0); // expected-warning{{zero as null pointer constant}}
}

// Warn on these too. Matches gcc and arguably makes sense.
void* pp = (decltype(nullptr))0; // expected-warning{{zero as null pointer constant}}
void* pp2 = static_cast<decltype(nullptr)>(0); // expected-warning{{zero as null pointer constant}}

// Shouldn't warn.
namespace pr34362 {
struct A { operator int*() { return nullptr; } };
void func() { if (nullptr == A()) {} }
void func2() { if ((nullptr) == A()) {} }
}
