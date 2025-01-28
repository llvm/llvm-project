// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux-gnu -fsyntax-only -std=c23 %s -pedantic -Wall

#include <limits.h>

enum us : unsigned short {
  us_max = USHRT_MAX,
  us_violation,  // expected-error {{enumerator value 65536 is not representable in the underlying type 'unsigned short'}}
  us_violation_2 = us_max + 1, // expected-error {{enumerator value is not representable in the underlying type 'unsigned short'}}
  us_wrap_around_to_zero = (unsigned short)(USHRT_MAX + 1) /* Okay: conversion
                            done in constant expression before conversion to
                            underlying type: unsigned semantics okay. */
};

enum ui : unsigned int {
  ui_max = UINT_MAX,
  ui_violation,  // expected-error {{enumerator value 4294967296 is not representable in the underlying type 'unsigned int'}}
  ui_no_violation = ui_max + 1,
  ui_wrap_around_to_zero = (unsigned int)(UINT_MAX + 1)
};

enum E1 : short;
enum E2 : short; // expected-note {{previous}}
enum E3; // expected-warning {{ISO C forbids forward references to 'enum' types}}
enum E4 : unsigned long long;

enum E1 : short { m11, m12 };
enum E1 x = m11;

enum E2 : long { // expected-error {{enumeration redeclared with different underlying type 'long' (was 'short')}}
  m21,
  m22
};

enum E3 { // expected-note {{definition of 'enum E3' is not complete until the closing '}'}}
          // expected-note@-1 {{previous}}
  m31,
  m32,
  m33 = sizeof(enum E3) // expected-error {{invalid application of 'sizeof' to an incomplete type 'enum E3'}}
};
enum E3 : int; // expected-error {{enumeration previously declared with nonfixed underlying type}}

enum E4 : unsigned long long {
  m40 = sizeof(enum E4),
  m41 = ULLONG_MAX,
  m42 // expected-error {{enumerator value 18446744073709551616 is not representable in the underlying type 'unsigned long long'}}
};

enum E5 y; // expected-error {{tentative definition has type 'enum E5' that is never completed}}
           // expected-warning@-1 {{ISO C forbids forward references to 'enum' types}}
           // expected-note@-2 {{forward declaration of 'enum E5'}}
enum E6 : long int z;   // expected-error {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}
enum E7 : long int = 0;  // expected-error {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}
                         // expected-error@-1 {{expected identifier or '('}}

enum underlying : unsigned char { b0 };

constexpr int a = _Generic(b0, int: 2, unsigned char: 1, default: 0);
constexpr int b = _Generic((enum underlying)b0, int: 2, unsigned char: 1, default: 0);
static_assert(a == 1);
static_assert(b == 1);

void f1(enum a : long b); // expected-error {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}
                          // expected-warning@-1 {{declaration of 'enum a' will not be visible outside of this function}}
void f2(enum c : long{x} d); // expected-warning {{declaration of 'enum c' will not be visible outside of this function}}
enum e : int f3(); // expected-error {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}

typedef enum t u; // expected-warning {{ISO C forbids forward references to 'enum' types}}
typedef enum v : short W; // expected-error {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}
typedef enum q : short { s } R;

struct s1 {
  int x;
  enum e:int : 1; // expected-error {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}
  int y;
};

enum forward; // expected-warning {{ISO C forbids forward references to 'enum' types}}
extern enum forward fwd_val0;  /* Constraint violation: incomplete type */
extern enum forward *fwd_ptr0; // expected-note {{previous}}
extern int
    *fwd_ptr0; // expected-error {{redeclaration of 'fwd_ptr0' with a different type: 'int *' vs 'enum forward *'}}

enum forward1 : int;
extern enum forward1 fwd_val1;
extern int fwd_val1;
extern enum forward1 *fwd_ptr1;
extern int *fwd_ptr1;

enum ee1 : short;
enum e : short f = 0; // expected-error {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}
enum g : short { yyy } h = yyy;

enum ee2 : typeof ((enum ee3 : short { A })0, (short)0);
