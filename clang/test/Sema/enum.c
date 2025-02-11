// RUN: %clang_cc1 -triple %itanium_abi_triple %s -fsyntax-only -verify=expected,pre-c23 -pedantic
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -fsyntax-only -std=c23 -verify=expected -pedantic
enum e {A,
        B = 42LL << 32,        // pre-c23-warning {{enumerator value which exceeds the range of 'int' is a C23 extension}}
      C = -4, D = 12456 };

enum f { a = -2147483648, b = 2147483647 }; // ok.

enum g {  // too negative
   c = -2147483649,         // pre-c23-warning {{enumerator value which exceeds the range of 'int' is a C23 extension}}
   d = 2147483647 };
enum h { e = -2147483648, // too pos
   f = 2147483648,           // pre-c23-warning {{enumerator value which exceeds the range of 'int' is a C23 extension}}
  i = 0xFFFF0000 // pre-c23-warning {{too large}}
};

// minll maxull
enum x                      // expected-warning {{enumeration values exceed range of largest integer}}
{ y = -9223372036854775807LL-1,  // pre-c23-warning {{enumerator value which exceeds the range of 'int' is a C23 extension}}
z = 9223372036854775808ULL };    // pre-c23-warning {{enumerator value which exceeds the range of 'int' is a C23 extension}}

int test(void) {
  return sizeof(enum e) ;
}

enum gccForwardEnumExtension ve; // expected-warning{{ISO C forbids forward references to 'enum' types}} \
// expected-error{{tentative definition has type 'enum gccForwardEnumExtension' that is never completed}} \
// expected-note{{forward declaration of 'enum gccForwardEnumExtension'}}

int test2(int i)
{
  ve + i; // expected-error{{invalid operands to binary expression}}
}

// PR2020
union u0;    // expected-note {{previous use is here}}
enum u0 { U0A }; // expected-error {{use of 'u0' with tag type that does not match previous declaration}}

extern enum some_undefined_enum ve2; // expected-warning {{ISO C forbids forward references to 'enum' types}}

void test4(void) {
  for (; ve2;) // expected-error {{statement requires expression of scalar type}}
    ;
  (_Bool)ve2;  // expected-error {{arithmetic or pointer type is required}}

  for (; ;ve2) // expected-warning {{expression result unused}}
    ;
  (void)ve2;
  ve2;         // expected-warning {{expression result unused}}
}

// PR2416
enum someenum {};  // expected-error {{use of empty enum}}

enum e0 { // expected-note {{previous definition is here}}
  E0 = sizeof(enum e0 { E1 }), // expected-error {{nested redefinition}}
};

// PR3173
enum { PR3173A, PR3173B = PR3173A+50 };

// PR2753
void foo(void) {
  enum xpto; // expected-warning{{ISO C forbids forward references to 'enum' types}}
  enum xpto; // expected-warning{{ISO C forbids forward references to 'enum' types}}
}

typedef enum { X = 0 }; // expected-warning{{typedef requires a name}}


enum NotYetComplete { // expected-note{{definition of 'enum NotYetComplete' is not complete until the closing '}'}}
  NYC1 = sizeof(enum NotYetComplete) // expected-error{{invalid application of 'sizeof' to an incomplete type 'enum NotYetComplete'}}
};

/// PR3688
struct s1 {
  enum e1 (*bar)(void); // expected-warning{{ISO C forbids forward references to 'enum' types}}
};

enum e1 { YES, NO };

static enum e1 badfunc(struct s1 *q) {
  return q->bar();
}


// Make sure we don't a.k.a. anonymous enums.
typedef enum {
  an_enumerator = 20
} an_enum;
char * s = (an_enum) an_enumerator; // expected-error {{incompatible integer to pointer conversion initializing 'char *' with an expression of type 'an_enum'}}

// PR4515
enum PR4515 {PR4515a=1u,PR4515b=(PR4515a-2)/2};
int CheckPR4515[PR4515b==0?1:-1];

// PR7911
extern enum PR7911T PR7911V; // expected-warning{{ISO C forbids forward references to 'enum' types}}
void PR7911F(void) {
  switch (PR7911V) // expected-error {{statement requires expression of integer type}}
    ;
}

char test5[__has_feature(enumerator_attributes) ? 1 : -1];

// PR8694
void PR8694(int* e) // expected-note {{passing argument to parameter 'e' here}}
{
}

void crash(enum E* e) // expected-warning {{declaration of 'enum E' will not be visible outside of this function}} \
                      // expected-warning {{ISO C forbids forward references to 'enum' types}}
{
        PR8694(e); // expected-warning {{incompatible pointer types passing 'enum E *' to parameter of type 'int *'}}
}

typedef enum { NegativeShort = (short)-1 } NegativeShortEnum;
int NegativeShortTest[NegativeShort == -1 ? 1 : -1];

// PR24610
enum Color { Red, Green, Blue }; // expected-note{{previous use is here}}
typedef struct Color NewColor; // expected-error {{use of 'Color' with tag type that does not match previous declaration}}

// Enumerations with a fixed underlying type. 
// https://github.com/llvm/llvm-project/issues/116880
#if __STDC_VERSION__ >= 202311L
  static_assert(__has_feature(c_fixed_enum));
  static_assert(__has_extension(c_fixed_enum)); // Matches behavior for c_alignas, etc
#else
  _Static_assert(__has_extension(c_fixed_enum), "");
  _Static_assert(!__has_feature(c_fixed_enum), "");
#if __STDC_VERSION__ < 201112L
  // expected-warning@-3 {{'_Static_assert' is a C11 extension}}
  // expected-warning@-3 {{'_Static_assert' is a C11 extension}}
#endif
#endif
typedef enum : unsigned char { Pink, Black, Cyan } Color; // pre-c23-warning {{enumeration types with a fixed underlying type are a C23 extension}}

// PR28903
// In C it is valid to define tags inside enums.
struct PR28903 {
  enum {
    PR28903_A = (enum {
      PR28903_B,
      PR28903_C = PR28903_B
    })0
  };
  int makeStructNonEmpty;
};

static int EnumRedecl; // expected-note 2 {{previous definition is here}}
struct S {
  enum {
    EnumRedecl = 4 // expected-error {{redefinition of 'EnumRedecl'}}
  } e;
};

union U {
  enum {
    EnumRedecl = 5 // expected-error {{redefinition of 'EnumRedecl'}}
  } e;
};

enum PR15071 {
  PR15071_One // expected-note {{previous definition is here}}
};

struct EnumRedeclStruct {
  enum {
    PR15071_One // expected-error {{redefinition of enumerator 'PR15071_One'}}
  } e;
};

enum struct GH42372_1 { // expected-error {{expected identifier or '{'}}
  One
};

// Because class is not a keyword in C, this looks like a forward declaration.
// expected-error@+4 {{expected ';' after top level declarator}}
// expected-error@+3 {{tentative definition has type 'enum class' that is never completed}}
// expected-warning@+2 {{ISO C forbids forward references to 'enum' types}}
// expected-note@+1 {{forward declaration of 'enum class'}}
enum class GH42372_2 {
  One
};

enum IncOverflow {
  V2 = __INT_MAX__,
  V3 // pre-c23-warning {{incremented enumerator value which exceeds the range of 'int' is a C23 extension}}
};

#if __STDC_VERSION__ >= 202311L
// FIXME: GCC picks __uint128_t as the underlying type for the enumeration
// value and Clang picks unsigned long long.
enum GH59352 { // expected-warning {{enumeration values exceed range of largest integer}}
 BigVal = 66666666666666666666wb
};
_Static_assert(BigVal == 66666666666666666666wb); /* expected-error {{static assertion failed due to requirement 'BigVal == 66666666666666666666wb'}}
                                                     expected-note {{expression evaluates to '11326434445538011818 == 66666666666666666666'}}
                                                   */
_Static_assert(
    _Generic(BigVal,                             // expected-error {{static assertion failed}}
    _BitInt(67) : 0,
    __INTMAX_TYPE__ : 0,
    __UINTMAX_TYPE__ : 0,
    long long : 0,
    unsigned long long : 0,
    __int128_t : 0,
    __uint128_t : 1
    )
);

#include <limits.h>

void fooinc23() {
  enum E1 {
    V1 = INT_MAX
  } e1;

  enum E2 {
    V2 = INT_MAX,
    V3
  } e2;

  enum E3 {
    V4 = INT_MAX,
    V5 = LONG_MIN
  } e3;

  enum E4 {
    V6 = 1u,
    V7 = 2wb
  } e4;

  _Static_assert(_Generic(V1, int : 1));
  _Static_assert(_Generic(V2, int : 0, unsigned int : 1));
  _Static_assert(_Generic(V3, int : 0, unsigned int : 1));
  _Static_assert(_Generic(V4, int : 0, signed long : 1));
  _Static_assert(_Generic(V5, int : 0, signed long : 1));
  _Static_assert(_Generic(V6, int : 1));
  _Static_assert(_Generic(V7, int : 1));
  _Static_assert(_Generic((enum E4){}, unsigned int : 1));

}

#endif // __STDC_VERSION__ >= 202311L
