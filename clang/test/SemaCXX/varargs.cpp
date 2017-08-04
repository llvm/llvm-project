// RUN: %clang_cc1 -std=c++03 -verify %s
// RUN: %clang_cc1 -std=c++11 -verify %s

__builtin_va_list ap;

class string;
void f(const string& s, ...) {  // expected-note {{parameter of type 'const string &' is declared here}}
  __builtin_va_start(ap, s); // expected-warning {{passing an object of reference type to 'va_start' has undefined behavior}}
}

void g(register int i, ...) {
  __builtin_va_start(ap, i); // UB in C, OK in C++
}

// Don't crash when there is no last parameter.
void no_params(...) {
  int a;
  __builtin_va_start(ap, a); // expected-warning {{second argument to 'va_start' is not the last named parameter}}
}

// Reject this. The __builtin_va_start would execute in Foo's non-variadic
// default ctor.
void record_context(int a, ...) {
  struct Foo {
    // expected-error@+1 {{'va_start' cannot be used outside a function}}
    void meth(int a, int b = (__builtin_va_start(ap, a), 0)) {}
  };
}

#if __cplusplus >= 201103L
// We used to have bugs identifying the correct enclosing function scope in a
// lambda.

void fixed_lambda_varargs_function(int a, ...) {
  [](int b) {
    __builtin_va_start(ap, b); // expected-error {{'va_start' used in function with fixed args}}
  }(42);
}
void varargs_lambda_fixed_function(int a) {
  [](int b, ...) {
    __builtin_va_start(ap, b); // correct
  }(42);
}

auto fixed_lambda_global = [](int f) {
  __builtin_va_start(ap, f); // expected-error {{'va_start' used in function with fixed args}}
};
auto varargs_lambda_global = [](int f, ...) {
  __builtin_va_start(ap, f); // correct
};

void record_member_init(int a, ...) {
  struct Foo {
    int a = 0;
    // expected-error@+1 {{'va_start' cannot be used outside a function}}
    int b = (__builtin_va_start(ap, a), 0);
  };
}
#endif
