// RUN: %clang_cc1 -triple x86_64-pc-linux-musl -fsyntax-only -verify=expected,nonglibc -Wno-strict-prototypes %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify -Wno-strict-prototypes %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-musl -fsyntax-only -verify=expected,nonglibc -x c++ -std=c++20 %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify -x c++ -std=c++20 %s

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

// In glibc environments, allow main-like signatures, but otherwise disallow
// any parameters.
void i1(int v) __attribute__((constructor)); // nonglibc-error {{'constructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
void j1(int v) __attribute__((destructor));  // nonglibc-error {{'destructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
void i2(int argc, char *argv[]) __attribute__((constructor)); // nonglibc-error {{'constructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
void j2(int argc, char *argv[]) __attribute__((destructor));  // nonglibc-error {{'destructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
void i3(int argc, char *const argv[], char *environ[]) __attribute__((constructor)); // nonglibc-error {{'constructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
void j3(int argc, const char *argv[], char *environ[]) __attribute__((destructor));  // nonglibc-error {{'destructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
void i4(int argc, float f) __attribute__((constructor)); // expected-error {{'constructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
void j4(int argc, float f) __attribute__((destructor));  // expected-error {{'destructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}

// Disallow calling conventions other than the C calling convention
__attribute__((regcall, constructor)) void k(void); // expected-error {{'constructor' attribute must be applied to a function with the C calling convention}}

#ifdef __cplusplus
// Disallow variadic functions.
__attribute__((constructor)) void g1(...); // expected-error {{'constructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
__attribute__((destructor)) void g2(...);  // expected-error {{'destructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}

struct S {
  // Not allowed on a nonstatic member function, but is allowed on a static
  // member function so long as it has no args/void return type.
  void mem1() __attribute__((constructor)); // expected-error {{'constructor' attribute cannot be applied to a member function}}
  void mem2() __attribute__((destructor));  // expected-error {{'destructor' attribute cannot be applied to a member function}}

  static signed nonmem1() __attribute__((constructor));
  static unsigned nonmem2() __attribute__((destructor));

  static _BitInt(32) nonmem3() __attribute__((constructor)); // expected-error {{'constructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
  static char nonmem4() __attribute__((destructor));         // expected-error {{'destructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}

  static void nonmem5(int) __attribute__((constructor)); // nonglibc-error {{'constructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
  static void nonmem6(int) __attribute__((destructor));  // nonglibc-error {{'destructor' attribute can only be applied to a function which accepts no arguments and has a 'void' or 'int' return type}}
};

consteval void consteval_func1() __attribute__((constructor)); // expected-error {{'constructor' attribute cannot be applied to a 'consteval' function}}
consteval void consteval_func2() __attribute__((destructor));  // expected-error {{'destructor' attribute cannot be applied to a 'consteval' function}}
#endif // __cplusplus

# 1 "source.c" 1 3
// Can use reserved priorities within a system header
void f(void) __attribute__((constructor(1)));
void f(void) __attribute__((destructor(1)));
# 1 "source.c" 2
