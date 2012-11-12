// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s

static int test0 __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}
static void test1() __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}

namespace test2 __attribute__((weak)) { // expected-warning {{'weak' attribute only applies to variables and functions}}
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

template <class T> struct Test7 {
  void test7() __attribute__((weak)) {}
};
namespace { class Internal; }
template struct Test7<Internal>;
template struct Test7<int>;
