// RUN: %clang_cc1 -fms-extensions -fsyntax-only -verify %s
// MSVC produces similar diagnostics.

__declspec(selectany) void foo() { } // expected-error{{'selectany' can only be applied to data items with external linkage}}

__declspec(selectany) int x1 = 1;

const __declspec(selectany) int x2 = 2; // expected-error{{'selectany' can only be applied to data items with external linkage}}

extern const __declspec(selectany) int x3 = 3;

extern const int x4;
const __declspec(selectany) int x4 = 4;

// MSDN says this is incorrect, but MSVC doesn't diagnose it.
extern __declspec(selectany) int x5;

static __declspec(selectany) int x6 = 2; // expected-error{{'selectany' can only be applied to data items with external linkage}}

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
