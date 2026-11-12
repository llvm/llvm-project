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
void func();                 // both-warning {{a function declaration without a prototype is deprecated in all versions of C and is treated as a zero-parameter prototype in C23, conflicting with a subsequent definition}} \
                                strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}
void func(a, b) int a, b; {} // both-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C23}}

void one_more(a, b) int a, b; {} // both-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C23}}

void sheesh(int a);
void sheesh(a) int a; {} // both-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C23}}

void another(); // strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}

int main(void) {
  another(1, 2);  // both-warning {{passing arguments to 'another' without a prototype is deprecated in all versions of C and is not supported in C23}}
}

void order1();        // both-warning {{a function declaration without a prototype is deprecated in all versions of C and is treated as a zero-parameter prototype in C23, conflicting with a subsequent declaration}} \
                         strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}
void order1(int i);   // both-note {{conflicting prototype is here}}

void order2(int i);   // both-note {{conflicting prototype is here}}
void order2();        // both-warning {{a function declaration without a prototype is deprecated in all versions of C and is treated as a zero-parameter prototype in C23, conflicting with a previous declaration}} \
                         strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}

void order3();        // both-warning {{a function declaration without a prototype is deprecated in all versions of C and is treated as a zero-parameter prototype in C23, conflicting with a subsequent definition}} \
                         strict-warning {{a function declaration without a prototype is deprecated in all versions of C}}
void order3(int i) {} // both-note {{conflicting prototype is here}}

// Just because the prototype is variadic doesn't mean we shouldn't warn on the
// K&R C function definition; this still changes behavior in C23.
void test(char*,...);
void test(fmt)        // both-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C23}}
        char*fmt;
{
}

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
  never_defined(1); // both-warning {{passing arguments to 'never_defined' without a prototype is deprecated in all versions of C and is not supported in C23}}

  // Ensure that calls to builtins without a traditional prototype are not
  // diagnosed.
  (void)__builtin_isless(1.0, 1.0); // OK

  // Calling a function whose prototype was provided by a function with an
  // identifier list is still fine.
  func(1, 2); // OK

  // Ensure that a call through a function pointer is still diagnosed properly.
  fp f;
  f(); // OK
  f(1, 2); // both-warning {{passing arguments to a function without a prototype is deprecated in all versions of C and is not supported in C23}}

  // Ensure that we don't diagnose when the diagnostic group is disabled.
  depr_non_prot(1); // OK
  strict_prot(1); // OK

  // Ensure we don't issue diagnostics if the function without a prototype was
  // later given a prototype by a definintion. Also ensure we don't duplicate
  // diagnostics if such a call is incorrect.
  func(1, 2); // OK
  func(1, 2, 3); // both-warning {{too many arguments in call to 'func'}}
}

// Issue 58181 -- we would issue the warning about the function without a
// prototype twice when the function was declared static in the following
// example.
static int GH58181(int x, int y);
static int GH58181(x, y) // both-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C23}}
    int x;
    int y;
{
    return x + y;
}
