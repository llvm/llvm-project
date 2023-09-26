// RUN: %clang_cc1 -fsyntax-only -verify -Wno-strict-prototypes %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ -std=c++20 %s

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

// Require a void or (unsigned) int return type
int g0(void) __attribute__((constructor));
signed int g1(void) __attribute__((constructor));
float g2(void) __attribute__((constructor)); // expected-error {{'constructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
int h0(void) __attribute__((destructor));
unsigned int h1(void) __attribute__((destructor));
float h2(void) __attribute__((destructor));  // expected-error {{'destructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}

// Require no parameters
void i(int v) __attribute__((constructor)); // expected-error {{'constructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
void j(int v) __attribute__((destructor));  // expected-error {{'destructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}

#ifdef __cplusplus
struct S {
  // Not allowed on a nonstatic member function, but is allowed on a static
  // member function so long as it has no args/void return type.
  void mem1() __attribute__((constructor)); // expected-error {{'constructor' attribute cannot be applied to a member function}}
  void mem2() __attribute__((destructor));  // expected-error {{'destructor' attribute cannot be applied to a member function}}

  static signed nonmem1() __attribute__((constructor));
  static unsigned nonmem2() __attribute__((destructor));

  static _BitInt(32) nonmem3() __attribute__((constructor)); // expected-error {{'constructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
  static char nonmem4() __attribute__((destructor));         // expected-error {{'destructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}

  static void nonmem5(int) __attribute__((constructor)); // expected-error {{'constructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
  static void nonmem6(int) __attribute__((destructor));  // expected-error {{'destructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
};

consteval void consteval_func1() __attribute__((constructor)); // expected-error {{'constructor' attribute cannot be applied to a 'consteval' function}}
consteval void consteval_func2() __attribute__((destructor));  // expected-error {{'destructor' attribute cannot be applied to a 'consteval' function}}
#endif // __cplusplus

# 1 "source.c" 1 3
// Can use reserved priorities within a system header
void f(void) __attribute__((constructor(1)));
void f(void) __attribute__((destructor(1)));
# 1 "source.c" 2
