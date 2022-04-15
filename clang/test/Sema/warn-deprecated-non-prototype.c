// RUN: %clang_cc1 -fsyntax-only -verify=both,expected %s
// RUN: %clang_cc1 -fsyntax-only -Wstrict-prototypes -verify=both,strict %s

// Test both with and without -Wstrict-prototypes because there are complicated
// interactions between it and -Wdeprecated-non-prototype.

// Off by default warnings, enabled by -pedantic or -Wstrict-prototypes
void other_func();   // strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}
void other_func() {} // strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}

void never_defined(); // strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}

typedef void (*fp)(); // strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}

void blerp(
  void (*func_ptr)() // strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}
);

void whatever(void) {
  extern void hoo_boy(); // strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}
}

void again() {} // strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}

// On by default warnings
void func();                 // expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}} \
                                strict-warning {{a function declaration without a prototype is deprecated in all versions of C}} \
                                strict-note {{a function declaration without a prototype is not supported in C2x}}
void func(a, b) int a, b; {} // both-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}}

void one_more(a, b) int a, b; {} // both-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}}

void sheesh(int a);
void sheesh(a) int a; {} // both-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}}

void another(); // strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}

int main(void) {
  another(1, 2);  // both-warning {{passing arguments to 'another' without a prototype is deprecated in all versions of C and is not supported in C2x}}
}

void order1();        // expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}} \
                         strict-warning {{a function declaration without a prototype is deprecated in all versions of C}} \
                         strict-note {{a function declaration without a prototype is not supported in C2x}}
void order1(int i);   // both-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}}

void order2(int i);
void order2();        // both-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}} \
                         strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}

void order3();        // expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}} \
                         strict-warning {{a function declaration without a prototype is deprecated in all versions of C}} \
                         strict-note {{a function declaration without a prototype is not supported in C2x}}
void order3(int i) {} // both-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}}

// Just because the prototype is variadic doesn't mean we shouldn't warn on the
// K&R C function definition; this still changes behavior in C2x.
void test(char*,...);
void test(fmt)        // both-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}}
        char*fmt;
{
}

// FIXME: we get two diagnostics here when running in pedantic mode. The first
// comes when forming the function type for the definition, and the second
// comes from merging the function declarations together. The second is the
// point at which we know the behavior has changed (because we can see the
// previous declaration at that point), but we've already issued the type
// error by that point. It's not ideal to be this chatty, but this situation
// should be pretty rare.
void blapp(int); // both-note {{previous declaration is here}}
void blapp() { } // both-error {{conflicting types for 'blapp'}} \
                 // strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}

// Disable -Wdeprecated-non-prototype
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-non-prototype"
void depr_non_prot(); // strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}
#pragma GCC diagnostic pop
// Reenable it.

// Disable -Wstrict-prototypes
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-prototypes"
void strict_prot(); // OK
#pragma GCC diagnostic pop
// Reenable it.

void calls(void) {
  // Ensure that we diagnose calls to functions without a prototype, but only
  // if they pass arguments.
  never_defined(); // OK
  never_defined(1); // both-warning {{passing arguments to 'never_defined' without a prototype is deprecated in all versions of C and is not supported in C2x}}

  // Ensure that calls to builtins without a traditional prototype are not
  // diagnosed.
  (void)__builtin_isless(1.0, 1.0); // OK

  // Calling a function whose prototype was provided by a function with an
  // identifier list is still fine.
  func(1, 2); // OK

  // Ensure that a call through a function pointer is still diagnosed properly.
  fp f;
  f(); // OK
  f(1, 2); // both-warning {{passing arguments to a function without a prototype is deprecated in all versions of C and is not supported in C2x}}

  // Ensure that we don't diagnose when the diagnostic group is disabled.
  depr_non_prot(1); // OK
  strict_prot(1); // OK

  // Ensure we don't issue diagnostics if the function without a prototype was
  // later given a prototype by a definintion. Also ensure we don't duplicate
  // diagnostics if such a call is incorrect.
  func(1, 2); // OK
  func(1, 2, 3); // both-warning {{too many arguments in call to 'func'}}
}
