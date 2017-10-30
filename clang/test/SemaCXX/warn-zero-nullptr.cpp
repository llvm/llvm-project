// RUN: %clang_cc1 -fsyntax-only -verify %s -isystem %S/Inputs -Wzero-as-null-pointer-constant -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify %s -isystem %S/Inputs -DSYSTEM_WARNINGS -Wzero-as-null-pointer-constant -Wsystem-headers -std=c++11

#include <warn-zero-nullptr.h>

#define MACRO (0)
#define MCRO(x) (x)

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

void f0(void* v = MACRO); // expected-warning{{zero as null pointer constant}}
void f1(void* v = NULL); // expected-warning{{zero as null pointer constant}}
void f2(void* v = MCRO(0)); // expected-warning{{zero as null pointer constant}}
void f3(void* v = MCRO(NULL)); // expected-warning{{zero as null pointer constant}}
void f4(void* v = 0); // expected-warning{{zero as null pointer constant}}
void f5(void* v);

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

template <typename T> void TmplFunc0(T var) {}
void Func0Test() {
  TmplFunc0<int>(0);
  TmplFunc0<int*>(0); // expected-warning {{zero as null pointer constant}}
  TmplFunc0<void*>(0); // expected-warning {{zero as null pointer constant}}
}

// FIXME: this one probably should not warn.
template <typename T> void TmplFunc1(int a, T default_value = 0) {} // expected-warning{{zero as null pointer constant}} expected-warning{{zero as null pointer constant}}
void FuncTest() {
  TmplFunc1<int>(0);
  TmplFunc1<int*>(0); // expected-note {{in instantiation of default function argument expression for 'TmplFunc1<int *>' required here}}
  TmplFunc1<void*>(0);  // expected-note {{in instantiation of default function argument expression for 'TmplFunc1<void *>' required here}}
}

template<typename T>
class TemplateClass0 {
 public:
  explicit TemplateClass0(T var) {}
};
void TemplateClass0Test() {
  TemplateClass0<int> a(0);
  TemplateClass0<int*> b(0); // expected-warning {{zero as null pointer constant}}
  TemplateClass0<void*> c(0); // expected-warning {{zero as null pointer constant}}
}

template<typename T>
class TemplateClass1 {
 public:
// FIXME: this one should *NOT* warn.
  explicit TemplateClass1(int a, T default_value = 0) {} // expected-warning{{zero as null pointer constant}} expected-warning{{zero as null pointer constant}}
};
void IgnoreSubstTemplateType1() {
  TemplateClass1<int> a(1);
  TemplateClass1<int*> b(1); // expected-note {{in instantiation of default function argument expression for 'TemplateClass1<int *>' required here}}
  TemplateClass1<void*> c(1); // expected-note {{in instantiation of default function argument expression for 'TemplateClass1<void *>' required here}}
}

#ifndef SYSTEM_WARNINGS
// Do not warn on *any* other macros from system headers, even if they
// expand to/their expansion contains NULL.
void* sys_init = SYSTEM_MACRO;
void* sys_init2 = OTHER_SYSTEM_MACRO;
#else
void* sys_init = SYSTEM_MACRO; // expected-warning {{zero as null pointer constant}}
void* sys_init2 = OTHER_SYSTEM_MACRO; // expected-warning {{zero as null pointer constant}}
#endif
