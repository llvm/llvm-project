// RUN: %clang_cc1 -fsyntax-only -verify=line0 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,line0 %s -DALL -Wgnu
// RUN: %clang_cc1 -fsyntax-only -verify=expected,line0 %s -DALL \
// RUN:   -Wgnu-zero-variadic-macro-arguments \
// RUN:   -Wgnu-zero-line-directive
// RUN: %clang_cc1 -fsyntax-only -verify %s -DNONE -Wgnu \
// RUN:   -Wno-gnu-zero-variadic-macro-arguments \
// RUN:   -Wno-gnu-zero-line-directive
// Additional disabled tests:
// %clang_cc1 -fsyntax-only -verify %s -DZEROARGS -Wgnu-zero-variadic-macro-arguments

#if NONE
// expected-no-diagnostics
#endif


#if ALL || ZEROARGS
// expected-warning@+9 {{passing no argument for the '...' parameter of a variadic macro is a C23 extension}}
// expected-note@+4 {{macro 'efoo' defined here}}
// expected-warning@+3 {{token pasting of ',' and '__VA_ARGS__' is a GNU extension}}
#endif

#define efoo(format, args...) foo(format , ##args)

void foo( const char* c )
{
  efoo("6");
}


#line 0 // line0-warning {{#line directive with zero argument is a GNU extension}}

// WARNING: Do not add more tests after the #line 0 line!  Add them before it.
