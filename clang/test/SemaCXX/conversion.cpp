// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only -Wconversion -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only -Wconversion %s 2>&1 | FileCheck %s

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
    A() : x(10) {} // expected-warning {{implicit truncation from 'int' to bitfield changes value from 10 to 2}}
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

  // we don't catch the case of #define FOO NULL ... int i = FOO; but that seems a bit narrow anyway
  // and avoiding that helps us skip these cases:
#define NULL_COND(cond) ((cond) ? &a : NULL)
  bool bl2 = NULL_COND(true); // don't warn on NULL conversion through the conditional operator across a macro boundary
  if (NULL_COND(true))
    ;
  while (NULL_COND(true))
    ;
  for (; NULL_COND(true); )
    ;
  do ;
  while(NULL_COND(true));
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
  void tmpl(char c = NULL, // expected-warning 4 {{implicit conversion of NULL constant to 'char'}}
            T a = NULL, // expected-warning {{implicit conversion of NULL constant to 'char'}} \
                           expected-warning 2 {{implicit conversion of NULL constant to 'int'}}
            T b = 1024) { // expected-warning {{implicit conversion from 'int' to 'char' changes value from 1024 to 0}}
  }

  template<typename T>
  void tmpl2(T t = NULL) {
  }

  void func() {
    tmpl<char>(); // expected-note 2 {{in instantiation of default function argument expression for 'tmpl<char>' required here}}
    tmpl<int>(); // expected-note 2 {{in instantiation of default function argument expression for 'tmpl<int>' required here}}
    // FIXME: We should warn only once for each template instantiation - not once for each call
    tmpl<int>(); // expected-note 2 {{in instantiation of default function argument expression for 'tmpl<int>' required here}}
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
