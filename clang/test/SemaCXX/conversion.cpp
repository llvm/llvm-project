// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only -Wconversion -std=c++11 -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only -Wconversion -std=c++11 %s 2>&1 | FileCheck %s

#include <stddef.h>

typedef   signed char  int8_t;
typedef   signed short int16_t;
typedef   signed int   int32_t;
typedef   signed long  int64_t;

typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int   uint32_t;
typedef unsigned long  uint64_t;

// <rdar://problem/7909130>
namespace test0 {
  int32_t test1_positive(char *I, char *E) {
    return (E - I); // expected-warning {{implicit conversion loses integer precision}}
  }

  int32_t test1_negative(char *I, char *E) {
    return static_cast<int32_t>(E - I);
  }

  uint32_t test2_positive(uint64_t x) {
    return x; // expected-warning {{implicit conversion loses integer precision}}
  }

  uint32_t test2_negative(uint64_t x) {
    return (uint32_t) x;
  }
}

namespace test1 {
  uint64_t test1(int x, unsigned y) {
    return sizeof(x == y);
  }

  uint64_t test2(int x, unsigned y) {
    return __alignof(x == y);
  }

  void * const foo();
  bool test2(void *p) {
    return p == foo();
  }
}

namespace test2 {
  struct A {
    unsigned int x : 2;
    A() : x(10) {} // expected-warning {{implicit truncation from 'int' to bit-field changes value from 10 to 2}}
  };
}

// This file tests -Wnull-conversion, a subcategory of -Wconversion
// which is on by default.

void test3() {
  int a = NULL; // expected-warning {{implicit conversion of NULL constant to 'int'}}
  int b;
  b = NULL; // expected-warning {{implicit conversion of NULL constant to 'int'}}
  long l = NULL; // FIXME: this should also warn, but currently does not if sizeof(NULL)==sizeof(inttype)
  int c = ((((NULL)))); // expected-warning {{implicit conversion of NULL constant to 'int'}}
  int d;
  d = ((((NULL)))); // expected-warning {{implicit conversion of NULL constant to 'int'}}
  bool bl = NULL; // expected-warning {{implicit conversion of NULL constant to 'bool'}}
  char ch = NULL; // expected-warning {{implicit conversion of NULL constant to 'char'}}
  unsigned char uch = NULL; // expected-warning {{implicit conversion of NULL constant to 'unsigned char'}}
  short sh = NULL; // expected-warning {{implicit conversion of NULL constant to 'short'}}
  double dbl = NULL; // expected-warning {{implicit conversion of NULL constant to 'double'}}

  // Use FileCheck to ensure we don't get any unnecessary macro-expansion notes 
  // (that don't appear as 'real' notes & can't be seen/tested by -verify)
  // CHECK-NOT: note:
  // CHECK: note: expanded from macro 'FINIT'
#define FINIT int a3 = NULL;
  FINIT // expected-warning {{implicit conversion of NULL constant to 'int'}}
  // we don't catch the case of #define FOO NULL ... int i = FOO; but that
  // seems a bit narrow anyway and avoiding that helps us skip other cases.

  int *ip = NULL;
  int (*fp)() = NULL;
  struct foo {
    int n;
    void func();
  };
  int foo::*datamem = NULL;
  int (foo::*funmem)() = NULL;
}

namespace test4 {
  // FIXME: We should warn for non-dependent args (only when the param type is also non-dependent) only once
  // not once for the template + once for every instantiation
  template<typename T>
  void tmpl(char c = NULL, // expected-warning 3 {{implicit conversion of NULL constant to 'char'}}
            T a = NULL, // expected-warning {{implicit conversion of NULL constant to 'char'}} \
                           expected-warning {{implicit conversion of NULL constant to 'int'}}
            T b = 1024) { // expected-warning {{implicit conversion from 'int' to 'char' changes value from 1024 to 0}}
  }

  template<typename T>
  void tmpl2(T t = NULL) {
  }

  void func() {
    tmpl<char>(); // expected-note 2 {{in instantiation of default function argument expression for 'tmpl<char>' required here}}
    tmpl<int>(); // expected-note 2 {{in instantiation of default function argument expression for 'tmpl<int>' required here}}
    tmpl<int>();
    tmpl2<int*>();
  }
}

namespace test5 {
  template<int I>
  void func() {
    bool b = I;
  }

  template void func<3>();
}

namespace test6 {
  decltype(nullptr) func() {
    return NULL;
  }
}

namespace test7 {
  bool fun() {
    bool x = nullptr; // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
    if (nullptr) {} // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
    return nullptr; // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  }
}

namespace test8 {
  #define NULL_COND(cond) ((cond) ? &num : NULL)
  #define NULL_WRAPPER NULL_COND(false)

  // don't warn on NULL conversion through the conditional operator across a
  // macro boundary
  void macro() {
    int num;
    bool b = NULL_COND(true);
    if (NULL_COND(true)) {}
    while (NULL_COND(true)) {}
    for (;NULL_COND(true);) {}
    do {} while (NULL_COND(true));

    if (NULL_WRAPPER) {}
    while (NULL_WRAPPER) {}
    for (;NULL_WRAPPER;) {}
    do {} while (NULL_WRAPPER);
  }

  // Identical to the previous function except with a template argument.
  // This ensures that template instantiation does not introduce any new
  // warnings.
  template <typename X>
  void template_and_macro() {
    int num;
    bool b = NULL_COND(true);
    if (NULL_COND(true)) {}
    while (NULL_COND(true)) {}
    for (;NULL_COND(true);) {}
    do {} while (NULL_COND(true));

    if (NULL_WRAPPER) {}
    while (NULL_WRAPPER) {}
    for (;NULL_WRAPPER;) {}
    do {} while (NULL_WRAPPER);
  }

  // Identical to the previous function except the template argument affects
  // the conditional statement.
  template <typename X>
  void template_and_macro2() {
    X num;
    bool b = NULL_COND(true);
    if (NULL_COND(true)) {}
    while (NULL_COND(true)) {}
    for (;NULL_COND(true);) {}
    do {} while (NULL_COND(true));

    if (NULL_WRAPPER) {}
    while (NULL_WRAPPER) {}
    for (;NULL_WRAPPER;) {}
    do {} while (NULL_WRAPPER);
  }

  void run() {
    template_and_macro<int>();
    template_and_macro<double>();
    template_and_macro2<int>();
    template_and_macro2<double>();
  }
}

// Don't warn on a nullptr to bool conversion when the nullptr is the return
// type of a function.
namespace test9 {
  typedef decltype(nullptr) nullptr_t;
  nullptr_t EXIT();

  bool test() {
    return EXIT();
  }
}

// Test NULL macro inside a macro has same warnings nullptr inside a macro.
namespace test10 {
#define test1(cond) \
      ((cond) ? nullptr : NULL)
#define test2(cond) \
      ((cond) ? NULL : nullptr)

#define assert(cond) \
      ((cond) ? foo() : bar())
  void foo();
  void bar();

  void run(int x) {
    if (test1(x)) {}
    if (test2(x)) {}
    assert(test1(x));
    assert(test2(x));
  }
}

namespace test11 {

#define assert11(expr) ((expr) ? 0 : 0)

// The whitespace in macro run1 are important to trigger the macro being split
// over multiple SLocEntry's.
#define run1() (dostuff() ? \
    NULL                                   : NULL)
#define run2() (dostuff() ? NULL : NULL)
int dostuff ();

void test(const char * content_type) {
  assert11(run1());
  assert11(run2());
}

}

namespace test12 {

#define x return NULL;

bool run() {
  x  // expected-warning{{}}
}

}

// More tests with macros.  Specficially, test function-like macros that either
// have a pointer return type or take pointer arguments.  Basically, if the
// macro was changed into a function and Clang doesn't warn, then it shouldn't
// warn for the macro either.
namespace test13 {
#define check_str_nullptr_13(str) ((str) ? str : nullptr)
#define check_str_null_13(str) ((str) ? str : NULL)
#define test13(condition) if (condition) return;
#define identity13(arg) arg
#define CHECK13(condition) test13(identity13(!(condition)))

void function1(const char* str) {
  CHECK13(check_str_nullptr_13(str));
  CHECK13(check_str_null_13(str));
}

bool some_bool_function(bool);
void function2() {
  CHECK13(some_bool_function(nullptr));  // expected-warning{{implicit conversion of nullptr constant to 'bool'}}
  CHECK13(some_bool_function(NULL));  // expected-warning{{implicit conversion of NULL constant to 'bool'}}
}

#define run_check_nullptr_13(str) \
    if (check_str_nullptr_13(str)) return;
#define run_check_null_13(str) \
    if (check_str_null_13(str)) return;
void function3(const char* str) {
  run_check_nullptr_13(str)
  run_check_null_13(str)
  if (check_str_nullptr_13(str)) return;
  if (check_str_null_13(str)) return;
}

void run(int* ptr);
#define conditional_run_13(ptr) \
    if (ptr) run(ptr);
void function4() {
  conditional_run_13(nullptr);
  conditional_run_13(NULL);
}
}
