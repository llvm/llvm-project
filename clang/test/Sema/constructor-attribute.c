// RUN: %clang_cc1 -fsyntax-only -verify -Wno-strict-prototypes %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s

int x1 __attribute__((constructor)); // expected-warning {{'constructor' attribute only applies to functions}}
void f(void) __attribute__((constructor));
void f(void) __attribute__((constructor(1)));   // expected-error {{'constructor' attribute requires integer constant between 101 and 65535 inclusive}}
void f(void) __attribute__((constructor(1,2))); // expected-error {{'constructor' attribute takes no more than 1 argument}}
void f(void) __attribute__((constructor(1.0))); // expected-error {{'constructor' attribute requires an integer constant}}
void f(void) __attribute__((constructor(0x100000000))); // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}
void f(void) __attribute__((constructor(101)));

int x2 __attribute__((destructor)); // expected-warning {{'destructor' attribute only applies to functions}}
void f(void) __attribute__((destructor));
void f(void) __attribute__((destructor(1)));   // expected-error {{'destructor' attribute requires integer constant between 101 and 65535 inclusive}}
void f(void) __attribute__((destructor(1,2))); // expected-error {{'destructor' attribute takes no more than 1 argument}}
void f(void) __attribute__((destructor(1.0))); // expected-error {{'destructor' attribute requires an integer constant}}
void f(void) __attribute__((destructor(101)));

void knr1() __attribute__((constructor));
void knr2() __attribute__((destructor));

// Require a void return type
int g(void) __attribute__((constructor)); // expected-error {{'constructor' attribute can only be applied to a function which accepts no arguments and has a 'void' return type}}
int h(void) __attribute__((destructor));  // expected-error {{'destructor' attribute can only be applied to a function which accepts no arguments and has a 'void' return type}}

// Require no parameters
void i(int v) __attribute__((constructor)); // expected-error {{'constructor' attribute can only be applied to a function which accepts no arguments and has a 'void' return type}}
void j(int v) __attribute__((destructor));  // expected-error {{'destructor' attribute can only be applied to a function which accepts no arguments and has a 'void' return type}}

#ifdef __cplusplus
struct S {
  // Not allowed on a nonstatic member function, but is allowed on a static
  // member function so long as it has no args/void return type.
  void mem1() __attribute__((constructor)); // expected-error {{'constructor' attribute cannot be applied to a member function}}
  void mem2() __attribute__((destructor));  // expected-error {{'destructor' attribute cannot be applied to a member function}}

  static void nonmem1() __attribute__((constructor));
  static void nonmem2() __attribute__((destructor));

  static int nonmem3() __attribute__((constructor)); // expected-error {{'constructor' attribute can only be applied to a function which accepts no arguments and has a 'void' return type}}
  static int nonmem4() __attribute__((destructor));  // expected-error {{'destructor' attribute can only be applied to a function which accepts no arguments and has a 'void' return type}}

  static void nonmem5(int) __attribute__((constructor)); // expected-error {{'constructor' attribute can only be applied to a function which accepts no arguments and has a 'void' return type}}
  static void nonmem6(int) __attribute__((destructor));  // expected-error {{'destructor' attribute can only be applied to a function which accepts no arguments and has a 'void' return type}}
};
#endif // __cplusplus

# 1 "source.c" 1 3
// Can use reserved priorities within a system header
void f(void) __attribute__((constructor(1)));
void f(void) __attribute__((destructor(1)));
# 1 "source.c" 2
