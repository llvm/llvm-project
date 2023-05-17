// RUN: %clang_cc1 -fms-compatibility -fsyntax-only -verify %s

// This should ignore the alignment and issue a warning about
// align not being used
auto func() -> __declspec(align(16)) int; // expected-warning{{attribute ignored when parsing type}}
static_assert(alignof(decltype(func())) == alignof(int), "error");

// The following should NOT assert since alignment should
// follow the type
struct Test { int a; };
using AlignedTest = __declspec(align(16)) const Test;
static_assert(alignof(AlignedTest) == 16, "error");

// Same here, no declaration to shift to
int i = (__declspec(align(16))int)12; // expected-warning{{attribute ignored when parsing type}}

// But there is a declaration here!
typedef __declspec(align(16)) int Foo;
static_assert(alignof(Foo) == 16, "error");

