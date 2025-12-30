// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -triple x86_64 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -triple i386 -fsyntax-only -verify -std=c++11 %s

struct {
  _BitInt(35) i : 33;
} x;
struct {
  _BitInt(35) i : 34;
} y;
_BitInt(33) xx{ x.i };
_BitInt(33) yy{ y.i };
// expected-error@-1 {{non-constant-expression cannot be narrowed from type '_BitInt(35)' to '_BitInt(33)' in initializer list}}
//   FIXME-expected-note@-2 {{insert an explicit cast to silence this issue}}

         _BitInt(2) S2 = 0;
unsigned _BitInt(2) U2 = 0;
         _BitInt(3) S3 = 0;
unsigned _BitInt(3) U3 = 0;

         _BitInt(2) bi0{ S2 };
         _BitInt(2) bi1{ U2 }; // expected-error {{non-constant-expression cannot be narrowed from type 'unsigned _BitInt(2)' to '_BitInt(2)' in initializer list}}
         _BitInt(2) bi2{ S3 }; // expected-error {{non-constant-expression cannot be narrowed from type '_BitInt(3)' to '_BitInt(2)' in initializer list}}
         _BitInt(2) bi3{ U3 }; // expected-error {{non-constant-expression cannot be narrowed from type 'unsigned _BitInt(3)' to '_BitInt(2)' in initializer list}}
unsigned _BitInt(2) bi4{ S2 }; // expected-error {{non-constant-expression cannot be narrowed from type '_BitInt(2)' to 'unsigned _BitInt(2)' in initializer list}}
unsigned _BitInt(2) bi5{ U2 };
unsigned _BitInt(2) bi6{ S3 }; // expected-error {{non-constant-expression cannot be narrowed from type '_BitInt(3)' to 'unsigned _BitInt(2)' in initializer list}}
unsigned _BitInt(2) bi7{ U3 }; // expected-error {{non-constant-expression cannot be narrowed from type 'unsigned _BitInt(3)' to 'unsigned _BitInt(2)' in initializer list}}
         _BitInt(3) bi8{ S2 };
         _BitInt(3) bi9{ U2 };
         _BitInt(3) bia{ S3 };
         _BitInt(3) bib{ U3 }; // expected-error {{non-constant-expression cannot be narrowed from type 'unsigned _BitInt(3)' to '_BitInt(3)' in initializer list}}
unsigned _BitInt(3) bic{ S2 }; // expected-error {{non-constant-expression cannot be narrowed from type '_BitInt(2)' to 'unsigned _BitInt(3)' in initializer list}}
unsigned _BitInt(3) bid{ U2 };
unsigned _BitInt(3) bie{ S3 }; // expected-error {{non-constant-expression cannot be narrowed from type '_BitInt(3)' to 'unsigned _BitInt(3)' in initializer list}}
unsigned _BitInt(3) bif{ U3 };
