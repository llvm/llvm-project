// RUN: %clang_cc1 -triple x86_64-win32 -fms-compatibility -fms-extensions -fsyntax-only -verify=expected -std=c++11 %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fms-compatibility -fms-extensions -fsyntax-only -verify=expected -std=c++11 %s
// RUN: %clang_cc1 -triple x86_64-win32-macho -fms-compatibility -fms-extensions -fsyntax-only -verify=expected,win23-macho -std=c++11 %s

// MSVC produces similar diagnostics.

__declspec(selectany) void foo() { } // expected-error{{'selectany' attribute only applies to variable declarations with external linkage}}

__declspec(selectany) int x1 = 1;

const __declspec(selectany) int x2 = 2; // expected-error{{'selectany' can only be applied to variables with external linkage}}

extern const __declspec(selectany) int x3 = 3;

extern const int x4;
const __declspec(selectany) int x4 = 4;

// MSDN says this is incorrect, but MSVC doesn't diagnose it.
extern __declspec(selectany) int x5;

static __declspec(selectany) int x6 = 2; // expected-error{{'selectany' can only be applied to variables with external linkage}}

// FIXME: MSVC accepts this and makes x7 externally visible and comdat, but keep
// it as internal and not weak/linkonce.
static int x7; // expected-note{{previous definition}}
extern __declspec(selectany) int x7;  // expected-warning{{attribute declaration must precede definition}}

int asdf() { return x7; }

class X {
 public:
  X(int i) { i++; };
  int i;
};

__declspec(selectany) X x(1);

namespace { class Internal {}; }
__declspec(selectany) auto x8 = Internal(); // expected-error {{'selectany' can only be applied to variables with external linkage}}


// The D3D11 headers do something like this.  MSVC doesn't error on this at
// all, even without the __declspec(selectany), in violation of the standard.
// We fall back to a warning for selectany to accept headers.
struct SomeStruct {
  int foo;
};
extern const __declspec(selectany) SomeStruct some_struct; // expected-warning {{default initialization of an object of const type 'const SomeStruct' without a user-provided default constructor is a Microsoft extension}}

// It should be possible to redeclare variables that were defined
// __declspec(selectany) previously.
extern const SomeStruct some_struct;

// Without selectany, this should stay an error.
const SomeStruct some_struct2; // expected-error {{default initialization of an object of const type 'const SomeStruct' without a user-provided default constructor}}

struct __declspec(selectany) S1 {}; // expected-error {{'selectany' attribute only applies to variable declarations with external linkage}}
__declspec(selectany) struct S1 s1;

void t() {
  __declspec(selectany) int a; // expected-error {{'selectany' can only be applied to variables with external linkage}}
  __declspec(selectany) extern int b;
  __declspec(selectany) static int c; // expected-error {{'selectany' can only be applied to variables with external linkage}}
  __declspec(selectany) thread_local int d; // expected-error {{'selectany' can only be applied to variables with external linkage}} win23-macho-error {{thread-local storage is not supported for the current target}}
}

struct S2 {};
struct __declspec(selectany) S2 s2; // expected-error {{'selectany' attribute only applies to variable declarations with external linkage}}

struct S3 {
  __declspec(selectany) static int a; // expected-error {{'selectany' can only be applied to variables with external linkage}}
};
