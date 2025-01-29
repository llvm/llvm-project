// RUN: %clang_cc1 -x c -fsyntax-only -Wparentheses -verify %s

// NOTE: Don't know if tests allow includes.
#include <stdbool.h>

// Do not emit the warning for compound-assignments. 
bool f(int x) { return x = 0; }  // expected-warning {{suggest parentheses around assignment used as truth value}}\
                                 // expected-note{{place parentheses around the assignment to silence this warning}}
bool f2(int x) { return x += 0; }

bool f3(bool x) { return x = 0; }

void test() {
  int x;

  // This should emit the `warn_condition_is_assignment` warning, since
  // C doesn't do implicit conversion booleans for conditions.
  if (x = 0) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{place parentheses around the assignment to silence this warning}}\
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  if (x = 4 && x){} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                    // expected-note{{place parentheses around the assignment to silence this warning}}\
                    // expected-note{{use '==' to turn this assignment into an equality comparison}}

  (void)(bool)(x = 1);
  (void)(bool)(int)(x = 1);


  bool _a = x = 3; // expected-warning {{suggest parentheses around assignment used as truth value}}\
                   // expected-note{{place parentheses around the assignment to silence this warning}}

  // Shouldn't warn for above cases if parentheses were provided.
  if ((x = 0)) {}
  bool _b = (x = 3);
}
