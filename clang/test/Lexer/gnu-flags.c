// RUN: %clang_cc1 -fsyntax-only -verify %s -DNONE
// RUN: %clang_cc1 -fsyntax-only -verify %s -DALL -Wgnu 
// RUN: %clang_cc1 -fsyntax-only -verify %s -DALL \
// RUN:   -Wgnu-zero-variadic-macro-arguments \
// RUN:   -Wgnu-imaginary-constant -Wgnu-binary-literal -Wgnu-zero-line-directive
// RUN: %clang_cc1 -fsyntax-only -verify %s -DNONE -Wgnu \
// RUN:   -Wno-gnu-zero-variadic-macro-arguments \
// RUN:   -Wno-gnu-imaginary-constant -Wno-gnu-binary-literal -Wno-gnu-zero-line-directive
// Additional disabled tests:
// %clang_cc1 -fsyntax-only -verify %s -DZEROARGS -Wgnu-zero-variadic-macro-arguments
// %clang_cc1 -fsyntax-only -verify %s -DIMAGINARYCONST -Wgnu-imaginary-constant
// %clang_cc1 -fsyntax-only -verify %s -DBINARYLITERAL -Wgnu-binary-literal
// %clang_cc1 -fsyntax-only -verify %s -DLINE0 -Wgnu-zero-line-directive

#if NONE
// expected-no-diagnostics
#endif


#if ALL || ZEROARGS
// expected-warning@+9 {{must specify at least one argument for '...' parameter of variadic macro}}
// expected-note@+4 {{macro 'efoo' defined here}}
// expected-warning@+3 {{token pasting of ',' and __VA_ARGS__ is a GNU extension}}
#endif

#define efoo(format, args...) foo(format , ##args)

void foo( const char* c )
{
  efoo("6");
}


#if ALL || IMAGINARYCONST
// expected-warning@+3 {{imaginary constants are a GNU extension}}
#endif

float _Complex c = 1.if;


#if ALL || BINARYLITERAL
// expected-warning@+3 {{binary integer literals are a GNU extension}}
#endif

int b = 0b0101;


// This case is handled differently because lit has a bug whereby #line 0 is reported to be on line 4294967295
// http://llvm.org/bugs/show_bug.cgi?id=16952
#if ALL || LINE0
#line 0 // expected-warning {{#line directive with zero argument is a GNU extension}}
#else
#line 0
#endif

// WARNING: Do not add more tests after the #line 0 line!  Add them before the LINE0 test
