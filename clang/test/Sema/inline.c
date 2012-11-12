// RUN: %clang_cc1 -fsyntax-only -verify %s

#if defined(INCLUDE)
// -------
// This section acts like a header file.
// -------

// Check the use of static variables in non-static inline functions.
static int staticVar; // expected-note + {{'staticVar' declared here}}
static int staticFunction(); // expected-note + {{'staticFunction' declared here}}
static struct { int x; } staticStruct; // expected-note + {{'staticStruct' declared here}}

inline int useStatic () { // expected-note 3 {{use 'static' to give inline function 'useStatic' internal linkage}}
  staticFunction(); // expected-warning{{static function 'staticFunction' is used in an inline function with external linkage}}
  (void)staticStruct.x; // expected-warning{{static variable 'staticStruct' is used in an inline function with external linkage}}
  return staticVar; // expected-warning{{static variable 'staticVar' is used in an inline function with external linkage}}
}

extern inline int useStaticFromExtern () { // no suggestions
  staticFunction(); // expected-warning{{static function 'staticFunction' is used in an inline function with external linkage}}
  return staticVar; // expected-warning{{static variable 'staticVar' is used in an inline function with external linkage}}
}

static inline int useStaticFromStatic () {
  staticFunction(); // no-warning
  return staticVar; // no-warning
}

extern inline int useStaticInlineFromExtern () {
  // Heuristic: if the function we're using is also inline, don't warn.
  // This can still be wrong (in this case, we end up inlining calls to
  // staticFunction and staticVar) but this got very noisy even using
  // standard headers.
  return useStaticFromStatic(); // no-warning
}

static int constFunction() __attribute__((const));

inline int useConst () {
  return constFunction(); // no-warning
}

#else
// -------
// This is the main source file.
// -------

#define INCLUDE
#include "inline.c"

// Check that we don't allow illegal uses of inline
inline int a; // expected-error{{'inline' can only appear on functions}}
typedef inline int b; // expected-error{{'inline' can only appear on functions}}
int d(inline int a); // expected-error{{'inline' can only appear on functions}}

// Check that the warnings from the "header file" aren't on by default in
// the main source file.

inline int useStaticMainFile () {
  staticFunction(); // no-warning
  return staticVar; // no-warning
}

// Check that the warnings show up when explicitly requested.

#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wstatic-in-inline"

inline int useStaticAgain () { // expected-note 2 {{use 'static' to give inline function 'useStaticAgain' internal linkage}}
  staticFunction(); // expected-warning{{static function 'staticFunction' is used in an inline function with external linkage}}
  return staticVar; // expected-warning{{static variable 'staticVar' is used in an inline function with external linkage}}
}

#pragma clang diagnostic pop

#endif


