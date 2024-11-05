// RUN: %clang_cc1 -x c++ -fsyntax-only -Wparentheses -verify %s
// RUN: %clang_cc1 -x c -fsyntax-only -Wparentheses -verify %s

// INFO: This warning is only issued for C and C++.
// This file is executed 3 times once for each language mention in the run-command.

#ifdef __cplusplus

// Do not emit the warning for compound-assignments. 
bool f(int x) { return x = 0; }  // expected-warning {{suggest parentheses around assignment used as truth value}}
bool f2(int x) { return x += 0; }

bool f3(bool x) { return x = 0; }

void test() {
  int x;
  if (x = 0) {} // expected-warning {{suggest parentheses around assignment used as truth value}}
  if (x = 4 && x){} // expected-warning {{suggest parentheses around assignment used as truth value}}

  (void)bool(x = 1); // expected-warning {{suggest parentheses around assignment used as truth value}}
  (void)(bool)(x = 1);

  // This should still emit since the RHS is casted to `int` before being casted back to `bool`.
  (void)bool(x = false); // expected-warning {{suggest parentheses around assignment used as truth value}}

  // Should only issue warning once, even if multiple implicit casts.
  // FIXME: This only checks that warning occurs not how often.
  (void)bool(bool(x = 1)); // expected-warning {{suggest parentheses around assignment used as truth value}}
  (void)bool(int(bool(x = 1))); // expected-warning {{suggest parentheses around assignment used as truth value}}
  (void)bool(int(x = 1));

  bool _a = x = 3; // expected-warning {{suggest parentheses around assignment used as truth value}}

  // Shouldn't warn for above cases if parentheses were provided.
  if ((x = 0)) {}
  (void)bool((x = 1));
  bool _b= (x = 3);
}

#elif defined(__OBJC__)

// NOTE: This warning shouldn't affect Objective-C
#import <Foundation/Foundation.h>

BOOL f(int x) { return x = 0; }
BOOL f2(int x) { return x += 0; }

BOOL f3(BOOL x) { return x = 0; }

void test() {
  int x;

  if (x = 0) {} // expected-warning {{using the result of an assignment as a condition without parentheses}}
  if (x = 4 && x){} // expected-warning {{using the result of an assignment as a condition without parentheses}}

  (void)(BOOL)(x = 1);
  (void)(BOOL)(int)(x = 1);

  BOOL _a = x = 3;

  if ((x = 0)) {}
  (void)BOOL((x = 1));
  BOOL _b= (x = 3);
}

#else

// NOTE: Don't know if tests allow includes.
#include <stdbool.h>

// Do not emit the warning for compound-assignments. 
bool f(int x) { return x = 0; }  // expected-warning {{suggest parentheses around assignment used as truth value}}
bool f2(int x) { return x += 0; }

bool f3(bool x) { return x = 0; }

void test() {
  int x;

  // This should emit the `warn_condition_is_assignment` warning since
  // C doesn't do implicit conversion booleans for conditions
  if (x = 0) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
                // expected-note{{place parentheses around the assignment to silence this warning}}
  if (x = 4 && x){} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                    // expected-note{{use '==' to turn this assignment into an equality comparison}} \
                    // expected-note{{place parentheses around the assignment to silence this warning}}

  (void)(bool)(x = 1);
  (void)(bool)(int)(x = 1);


  bool _a = x = 3; // expected-warning {{suggest parentheses around assignment used as truth value}}

  // Shouldn't warn for above cases if parentheses were provided.
  if ((x = 0)) {}
  bool _b = (x = 3);
}

#endif
