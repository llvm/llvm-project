// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic
enum e {A, 
        B = 42LL << 32,        // expected-warning {{ISO C restricts enumerator values to range of 'int'}}
      C = -4, D = 12456 };

enum f { a = -2147483648, b = 2147483647 }; // ok.

enum g {  // too negative
   c = -2147483649,         // expected-warning {{ISO C restricts enumerator values to range of 'int'}}
   d = 2147483647 };
enum h { e = -2147483648, // too pos
   f = 2147483648,           // expected-warning {{ISO C restricts enumerator values to range of 'int'}}
  i = 0xFFFF0000 // expected-warning {{too large}}
}; 

// minll maxull
enum x                      // expected-warning {{enumeration values exceed range of largest integer}}
{ y = -9223372036854775807LL-1,  // expected-warning {{ISO C restricts enumerator values to range of 'int'}}
z = 9223372036854775808ULL };    // expected-warning {{ISO C restricts enumerator values to range of 'int'}}

int test() {
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


// rdar://6095136
extern enum some_undefined_enum ve2; // expected-warning {{ISO C forbids forward references to 'enum' types}}

void test4() {
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

// <rdar://problem/6093889>
enum e0 { // expected-note {{previous definition is here}}
  E0 = sizeof(enum e0 { E1 }), // expected-error {{nested redefinition}}
};

// PR3173
enum { PR3173A, PR3173B = PR3173A+50 };

// PR2753
void foo() {
  enum xpto; // expected-warning{{ISO C forbids forward references to 'enum' types}}
  enum xpto; // expected-warning{{ISO C forbids forward references to 'enum' types}}
}

// <rdar://problem/6503878>
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
char * s = (an_enum) an_enumerator; // expected-warning {{incompatible integer to pointer conversion initializing 'char *' with an expression of type 'an_enum'}}

// PR4515
enum PR4515 {PR4515a=1u,PR4515b=(PR4515a-2)/2};
int CheckPR4515[PR4515b==0?1:-1];

// PR7911
extern enum PR7911T PR7911V; // expected-warning{{ISO C forbids forward references to 'enum' types}}
void PR7911F() {
  switch (PR7911V); // expected-error {{statement requires expression of integer type}}
}

char test5[__has_feature(enumerator_attributes) ? 1 : -1];

// PR8694
// rdar://8707031
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
