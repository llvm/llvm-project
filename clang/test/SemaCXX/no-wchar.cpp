// RUN: %clang_cc1 -triple i386-pc-win32 -fsyntax-only -fno-wchar -verify %s
wchar_t x; // expected-error {{unknown type name 'wchar_t'}}

typedef unsigned short wchar_t;
void foo(const wchar_t* x);

void bar() {
  foo(L"wide string literal");
}

void foo1(wchar_t * t = L"");
// expected-warning@-1 {{conversion from string literal to 'wchar_t *' (aka 'unsigned short *') is deprecated}}

short *a = L"";
// expected-error@-1 {{cannot initialize a variable of type 'short *' with an lvalue of type 'const unsigned short [1]'}}
char *b = L"";
// expected-error@-1 {{cannot initialize a variable of type 'char *' with an lvalue of type 'const unsigned short [1]'}}

// NOTE: MSVC allows deprecated conversion in conditional expression if at least
// one of the operand is a string literal but Clang doesn't allow it.
wchar_t *c = true ? L"a" : L"";
// expected-error@-1 {{cannot initialize a variable of type 'wchar_t *' (aka 'unsigned short *') with}}

const wchar_t *d1 = 0;
const wchar_t *d2 = 0;
wchar_t *d = true ? d1 : d2;
// expected-error@-1 {{cannot initialize a variable of type 'wchar_t *' (aka 'unsigned short *') with}}

wchar_t* e = (const wchar_t*)L"";
// expected-error@-1 {{cannot initialize a variable of type 'wchar_t *' (aka 'unsigned short *') with an rvalue of type 'const wchar_t *' (aka 'const unsigned short *')}}
