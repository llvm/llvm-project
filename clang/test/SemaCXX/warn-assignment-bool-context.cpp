// RUN: %clang_cc1 -x c++ -fsyntax-only -Wparentheses -verify %s

// Do not emit the warning for compound-assignments. 
bool f(int x) { return x = 0; }  // expected-warning {{using the result of an assignment as a truth value without parentheses}} \
                                 // expected-note{{place parentheses around the assignment to silence this warning}}
bool f2(int x) { return x += 0; }

bool f3(bool x) { return x = 0; }

void test() {
  int x;

  // Assignments inside of conditions should still emit the more specific `==` fixits.
  if (x = 0) {} // expected-warning {{using the result of an assignment as a truth value without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
                // expected-note{{place parentheses around the assignment to silence this warning}}
  if (x = 4 && x){} // expected-warning {{using the result of an assignment as a truth value without parentheses}} \
                    // expected-note{{use '==' to turn this assignment into an equality comparison}} \
                    // expected-note{{place parentheses around the assignment to silence this warning}}


  (void)bool(x = 1); // expected-warning {{using the result of an assignment as a truth value without parentheses}}\
                     // expected-note{{place parentheses around the assignment to silence this warning}}
  (void)(bool)(x = 1);

  // This should still emit since the RHS is casted to `int` before being casted back to `bool`.
  (void)bool(x = false); // expected-warning {{using the result of an assignment as a truth value without parentheses}} \
                         // expected-note{{place parentheses around the assignment to silence this warning}}

  // Should only issue warning once, even if multiple implicit casts.
  // FIXME: This only checks that warning occurs not how often.
  (void)bool(bool(x = 1)); // expected-warning {{using the result of an assignment as a truth value without parentheses}} \
                           // expected-note{{place parentheses around the assignment to silence this warning}}
  (void)bool(int(bool(x = 1))); // expected-warning {{using the result of an assignment as a truth value without parentheses}} \
                                // expected-note{{place parentheses around the assignment to silence this warning}}
  (void)bool(int(x = 1));

  bool _a = x = 3; // expected-warning {{using the result of an assignment as a truth value without parentheses}} \
                   // expected-note{{place parentheses around the assignment to silence this warning}}

  // Shouldn't warn for above cases if parentheses were provided.
  if ((x = 0)) {}
  (void)bool((x = 1));
  bool _b= (x = 3);
}
