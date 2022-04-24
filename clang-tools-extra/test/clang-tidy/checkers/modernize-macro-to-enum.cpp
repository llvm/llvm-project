// RUN: %check_clang_tidy -std=c++14-or-later %s modernize-macro-to-enum %t -- -- -I%S/Inputs/modernize-macro-to-enum -fno-delayed-template-parsing
// C++14 or later required for binary literals.

#if 1
#include "modernize-macro-to-enum.h"

// These macros are skipped due to being inside a conditional compilation block.
#define GOO_RED 1
#define GOO_GREEN 2
#define GOO_BLUE 3

#endif

#define RED 0xFF0000
#define GREEN 0x00FF00
#define BLUE 0x0000FF
// CHECK-MESSAGES: :[[@LINE-3]]:1: warning: replace macro with enum [modernize-macro-to-enum]
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'RED' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'GREEN' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'BLUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: RED = 0xFF0000,
// CHECK-FIXES-NEXT: GREEN = 0x00FF00,
// CHECK-FIXES-NEXT: BLUE = 0x0000FF
// CHECK-FIXES-NEXT: };

// Verify that comments are preserved.
#define CoordModeOrigin         0   /* relative to the origin */
#define CoordModePrevious       1   /* relative to previous point */
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-3]]:9: warning: macro 'CoordModeOrigin' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-3]]:9: warning: macro 'CoordModePrevious' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: CoordModeOrigin =         0,   /* relative to the origin */
// CHECK-FIXES-NEXT: CoordModePrevious =       1   /* relative to previous point */
// CHECK-FIXES-NEXT: };

// Verify that multiline comments are preserved.
#define BadDrawable         9   /* parameter not a Pixmap or Window */
#define BadAccess           10  /* depending on context:
                                - key/button already grabbed
                                - attempt to free an illegal 
                                  cmap entry 
                                - attempt to store into a read-only 
                                  color map entry. */
                                // - attempt to modify the access control
                                //   list from other than the local host.
                                //
#define BadAlloc            11  /* insufficient resources */
// CHECK-MESSAGES: :[[@LINE-11]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-12]]:9: warning: macro 'BadDrawable' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-12]]:9: warning: macro 'BadAccess' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: macro 'BadAlloc' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: BadDrawable =         9,   /* parameter not a Pixmap or Window */
// CHECK-FIXES-NEXT: BadAccess =           10,  /* depending on context:
// CHECK-FIXES-NEXT:                                 - key/button already grabbed
// CHECK-FIXES-NEXT:                                 - attempt to free an illegal 
// CHECK-FIXES-NEXT:                                   cmap entry 
// CHECK-FIXES-NEXT:                                 - attempt to store into a read-only 
// CHECK-FIXES-NEXT:                                   color map entry. */
// CHECK-FIXES-NEXT:                                 // - attempt to modify the access control
// CHECK-FIXES-NEXT:                                 //   list from other than the local host.
// CHECK-FIXES-NEXT:                                 //
// CHECK-FIXES-NEXT: BadAlloc =            11  /* insufficient resources */
// CHECK-FIXES-NEXT: };

// Undefining a macro invalidates adjacent macros
// from being considered as an enum.
#define REMOVED1 1
#define REMOVED2 2
#define REMOVED3 3
#undef REMOVED2
#define VALID1 1
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'VALID1' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: VALID1 = 1
// CHECK-FIXES-NEXT: };

#define UNDEF1 1
#define UNDEF2 2
#define UNDEF3 3

// Undefining a macro later invalidates the set of possible adjacent macros
// from being considered as an enum.
#undef UNDEF2

// Integral constants can have an optional sign
#define SIGNED1 +1
#define SIGNED2 -1
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-3]]:9: warning: macro 'SIGNED1' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-3]]:9: warning: macro 'SIGNED2' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: SIGNED1 = +1,
// CHECK-FIXES-NEXT: SIGNED2 = -1
// CHECK-FIXES-NEXT: };

// Integral constants with bitwise negated values
#define UNOP1 ~0U
#define UNOP2 ~1U
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-3]]:9: warning: macro 'UNOP1' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-3]]:9: warning: macro 'UNOP2' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: UNOP1 = ~0U,
// CHECK-FIXES-NEXT: UNOP2 = ~1U
// CHECK-FIXES-NEXT: };

// Integral constants in other bases and with suffixes are OK
#define BASE1 0777    // octal
#define BASE2 0xDEAD  // hexadecimal
#define BASE3 0b0011  // binary
#define SUFFIX1 +1U
#define SUFFIX2 -1L
#define SUFFIX3 +1UL
#define SUFFIX4 -1LL
#define SUFFIX5 +1ULL
// CHECK-MESSAGES: :[[@LINE-8]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-9]]:9: warning: macro 'BASE1' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-9]]:9: warning: macro 'BASE2' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-9]]:9: warning: macro 'BASE3' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-9]]:9: warning: macro 'SUFFIX1' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-9]]:9: warning: macro 'SUFFIX2' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-9]]:9: warning: macro 'SUFFIX3' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-9]]:9: warning: macro 'SUFFIX4' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-9]]:9: warning: macro 'SUFFIX5' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: BASE1 = 0777,    // octal
// CHECK-FIXES-NEXT: BASE2 = 0xDEAD,  // hexadecimal
// CHECK-FIXES-NEXT: BASE3 = 0b0011,  // binary
// CHECK-FIXES-NEXT: SUFFIX1 = +1U,
// CHECK-FIXES-NEXT: SUFFIX2 = -1L,
// CHECK-FIXES-NEXT: SUFFIX3 = +1UL,
// CHECK-FIXES-NEXT: SUFFIX4 = -1LL,
// CHECK-FIXES-NEXT: SUFFIX5 = +1ULL
// CHECK-FIXES-NEXT: };

// A limited form of constant expression is recognized: a parenthesized
// literal or a parenthesized literal with the unary operators +, - or ~.
#define PAREN1 (-1)
#define PAREN2 (1)
#define PAREN3 (+1)
#define PAREN4 (~1)
// CHECK-MESSAGES: :[[@LINE-4]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-5]]:9: warning: macro 'PAREN1' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-5]]:9: warning: macro 'PAREN2' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-5]]:9: warning: macro 'PAREN3' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-5]]:9: warning: macro 'PAREN4' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: PAREN1 = (-1),
// CHECK-FIXES-NEXT: PAREN2 = (1),
// CHECK-FIXES-NEXT: PAREN3 = (+1),
// CHECK-FIXES-NEXT: PAREN4 = (~1)
// CHECK-FIXES-NEXT: };

// More complicated parenthesized expressions are excluded.
// Expansions that are not surrounded by parentheses are excluded.
// Nested matching parentheses are stripped.
#define COMPLEX_PAREN1 (x+1)
#define COMPLEX_PAREN2 (x+1
#define COMPLEX_PAREN3 (())
#define COMPLEX_PAREN4 ()
#define COMPLEX_PAREN5 (+1)
#define COMPLEX_PAREN6 ((+1))
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-3]]:9: warning: macro 'COMPLEX_PAREN5' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-3]]:9: warning: macro 'COMPLEX_PAREN6' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: COMPLEX_PAREN5 = (+1),
// CHECK-FIXES-NEXT: COMPLEX_PAREN6 = ((+1))
// CHECK-FIXES-NEXT: };

// Macros appearing in conditional expressions can't be replaced
// by enums.
#define USE_FOO 1
#define USE_BAR 0
#define USE_IF 1
#define USE_ELIF 1
#define USE_IFDEF 1
#define USE_IFNDEF 1

// Undef'ing first and then defining later should still exclude this macro
#undef USE_UINT64
#define USE_UINT64 0
#undef USE_INT64
#define USE_INT64 0

#if defined(USE_FOO) && USE_FOO
extern void foo();
#else
inline void foo() {}
#endif

#if USE_BAR
extern void bar();
#else
inline void bar() {}
#endif

#if USE_IF
inline void used_if() {}
#endif

#if 0
#elif USE_ELIF
inline void used_elif() {}
#endif

#ifdef USE_IFDEF
inline void used_ifdef() {}
#endif

#ifndef USE_IFNDEF
#else
inline void used_ifndef() {}
#endif

// Regular conditional compilation blocks should leave previous
// macro enums alone.
#if 0
#include <non-existent.h>
#endif

// Conditional compilation blocks invalidate adjacent macros
// from being considered as an enum.  Conditionally compiled
// blocks could contain macros that should rightly be included
// in the enum, but we can't explore multiple branches of a
// conditionally compiled section in clang-tidy, only the active
// branch based on compilation options.
#define CONDITION1 1
#define CONDITION2 2
#if 0
#define CONDITION3 3
#else
#define CONDITION3 -3
#endif

#define IFDEF1 1
#define IFDEF2 2
#ifdef FROB
#define IFDEF3 3
#endif

#define IFNDEF1 1
#define IFNDEF2 2
#ifndef GOINK
#define IFNDEF3 3
#endif

// Macros used in conditions are invalidated, even if they look
// like enums after they are used in conditions.
#if DEFINED_LATER1
#endif
#ifdef DEFINED_LATER2
#endif
#ifndef DEFINED_LATER3
#endif
#undef DEFINED_LATER4
#if ((defined(DEFINED_LATER5) || DEFINED_LATER6) && DEFINED_LATER7) || (DEFINED_LATER8 > 10)
#endif

#define DEFINED_LATER1 1
#define DEFINED_LATER2 2
#define DEFINED_LATER3 3
#define DEFINED_LATER4 4
#define DEFINED_LATER5 5
#define DEFINED_LATER6 6
#define DEFINED_LATER7 7
#define DEFINED_LATER8 8

// Sometimes an argument to ifdef can be classified as a keyword token.
#ifdef __restrict
#endif

// These macros do not expand to integral constants.
#define HELLO "Hello, "
#define WORLD "World"
#define EPS1 1.0F
#define EPS2 1e5
#define EPS3 1.

#define DO_RED draw(RED)
#define DO_GREEN draw(GREEN)
#define DO_BLUE draw(BLUE)

#define FN_RED(x) draw(RED | x)
#define FN_GREEN(x) draw(GREEN | x)
#define FN_BLUE(x) draw(BLUE | x)

extern void draw(unsigned int Color);

void f()
{
  draw(RED);
  draw(GREEN);
  draw(BLUE);
  DO_RED;
  DO_GREEN;
  DO_BLUE;
  FN_RED(0);
  FN_GREEN(0);
  FN_BLUE(0);
}

// Ignore macros defined inside a top-level function definition.
void g(int x)
{
  if (x != 0) {
#define INSIDE1 1
#define INSIDE2 2
    if (INSIDE1 > 1) {
      f();
    }
  } else {
    if (INSIDE2 == 1) {
      f();
    }
  }
}

// Ignore macros defined inside a top-level function declaration.
extern void g2(
#define INSIDE3 3
#define INSIDE4 4
);

// Ignore macros defined inside a record (structure) declaration.
struct S {
#define INSIDE5 5
#define INSIDE6 6
  char storage[INSIDE5];
};
class C {
#define INSIDE7 7
#define INSIDE8 8
};

// Ignore macros defined inside a template function definition.
template <int N>
#define INSIDE9 9
bool fn()
{
  #define INSIDE10 10
  return INSIDE9 > 1 || INSIDE10 < N;
}

// Ignore macros defined inside a variable declaration.
extern int
#define INSIDE11 11
v;

// Ignore macros defined inside a template class definition.
template <int N>
class C2 {
public:
#define INSIDE12 12
    char storage[N];
  bool f() {
    return N > INSIDE12;
  }
  bool g();
};

// Ignore macros defined inside a template member function definition.
template <int N>
#define INSIDE13 13
bool C2<N>::g() {
#define INSIDE14 14
  return N < INSIDE12 || N > INSIDE13 || INSIDE14 > N;
};

// Ignore macros defined inside a template type alias.
template <typename T>
class C3 {
  T data;
};
template <typename T>
#define INSIDE15 15
using Data = C3<T[INSIDE15]>;

// Ignore macros defined inside a type alias.
using Data2 =
#define INSIDE16 16
    char[INSIDE16];

// Ignore macros defined inside a (constexpr) variable definition.
constexpr int
#define INSIDE17 17
value = INSIDE17;
