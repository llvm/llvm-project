// RUN: %clang_cc1 -x c -fsyntax-only -Wparentheses -verify %s

#define bool _Bool
#define true 1
#define false 0

// Do not emit the warning for compound-assignments. 
bool f(int x) { return x = 0; }  // expected-warning {{using the result of an assignment as a truth value without parentheses}}\
                                 // expected-note{{place parentheses around the assignment to silence this warning}}
bool f2(int x) { return x += 0; }

bool f3(bool x) { return x = 0; }

void test() {
  int x;

  // This should emit the `warn_assignment_bool_context` warning once, since
  // C doesn't do implicit conversion booleans for conditions.
  if (x = 0) {} // expected-warning {{using the result of an assignment as a truth value without parentheses}} \
                // expected-note{{place parentheses around the assignment to silence this warning}}\
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  if (x = 4 && x){} // expected-warning {{using the result of an assignment as a truth value without parentheses}} \
                    // expected-note{{place parentheses around the assignment to silence this warning}}\
                    // expected-note{{use '==' to turn this assignment into an equality comparison}}

  (void)(bool)(x = 1);
  (void)(bool)(int)(x = 1);


  bool _a = x = 3; // expected-warning {{using the result of an assignment as a truth value without parentheses}}\
                   // expected-note{{place parentheses around the assignment to silence this warning}}

  // Shouldn't warn for above cases if parentheses were provided.
  if ((x = 0)) {}
  bool _b = (x = 3);
}
