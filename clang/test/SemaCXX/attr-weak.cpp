// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify -std=c++11 %s

static int test0 __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}
static void test1() __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}

namespace test2 __attribute__((weak)) { // expected-warning {{'weak' attribute only applies to variables, functions, and classes}}
}

namespace {
  int test3 __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}
  void test4() __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}
}

struct Test5 {
  static void test5() __attribute__((weak)); // no error
};

namespace {
  struct Test6 {
    static void test6() __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}
  };
}

// GCC rejects the instantiation with the internal type, but some existing
// code expects it. It is also not that different from giving hidden visibility
// to parts of a template that have explicit default visibility, so we accept
// this.
template <class T> struct Test7 {
  void test7() __attribute__((weak)) {}
  static int var __attribute__((weak));
};
template <class T>
int Test7<T>::var;
namespace { class Internal {}; }
template struct Test7<Internal>;
template struct Test7<int>;

class __attribute__((weak)) Test8 {}; // OK

__attribute__((weak)) auto Test9 = Internal(); // expected-error {{weak declaration cannot have internal linkage}}
