// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -verify %s

#if defined(INCLUDE)
// -------
// This section acts like a header file.
// -------

// Check the use of static variables in non-static inline functions.
static int staticVar; // expected-note + {{'staticVar' declared here}}
static int staticFunction(void); // expected-note + {{'staticFunction' declared here}}
static struct { int x; } staticStruct; // expected-note + {{'staticStruct' declared here}}

inline int useStatic (void) { // expected-note 3 {{use 'static' to give inline function 'useStatic' internal linkage}}
  staticFunction(); // expected-warning{{using static function 'staticFunction' in an inline function with external linkage is a C2y extension}}
  (void)staticStruct.x; // expected-warning{{using static variable 'staticStruct' in an inline function with external linkage is a C2y extension}}
  return staticVar; // expected-warning{{using static variable 'staticVar' in an inline function with external linkage is a C2y extension}}
}

extern inline int useStaticFromExtern (void) { // no suggestions
  staticFunction(); // expected-warning{{using static function 'staticFunction' in an inline function with external linkage is a C2y extension}}
  return staticVar; // expected-warning{{using static variable 'staticVar' in an inline function with external linkage is a C2y extension}}
}

static inline int useStaticFromStatic (void) {
  staticFunction(); // no-warning
  return staticVar; // no-warning
}

extern inline int useStaticInlineFromExtern (void) {
  // Heuristic: if the function we're using is also inline, don't warn.
  // This can still be wrong (in this case, we end up inlining calls to
  // staticFunction and staticVar) but this got very noisy even using
  // standard headers.
  return useStaticFromStatic(); // no-warning
}

static int constFunction(void) __attribute__((const));

inline int useConst (void) {
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

inline int useStaticMainFile (void) {
  staticFunction(); // no-warning
  return staticVar; // no-warning
}

// Check that the warnings show up when explicitly requested.

#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wstatic-in-inline"

inline int useStaticAgain (void) { // expected-note 2 {{use 'static' to give inline function 'useStaticAgain' internal linkage}}
  staticFunction(); // expected-warning{{using static function 'staticFunction' in an inline function with external linkage is a C2y extension}}
  return staticVar; // expected-warning{{using static variable 'staticVar' in an inline function with external linkage is a C2y extension}}
}

#pragma clang diagnostic pop

inline void defineStaticVar(void) { // expected-note {{use 'static' to give inline function 'defineStaticVar' internal linkage}}
  static const int x = 0; // ok
  static int y = 0; // expected-warning {{non-constant static local variable in inline function may be different in different files}}
}

extern inline void defineStaticVarInExtern(void) {
  static const int x = 0; // ok
  static int y = 0; // ok
}

// Check behavior of line markers.
# 1 "XXX.h" 1
inline int useStaticMainFileInLineMarker(void) { // expected-note 2 {{use 'static' to give inline function 'useStaticMainFileInLineMarker' internal linkage}}
  staticFunction(); // expected-warning{{using static function 'staticFunction' in an inline function with external linkage is a C2y extension}}
  return staticVar; // expected-warning{{using static variable 'staticVar' in an inline function with external linkage is a C2y extension}}
}
# 100 "inline.c" 2

inline int useStaticMainFileAfterLineMarker(void) {
  staticFunction(); // no-warning
  return staticVar; // no-warning
}

#endif


