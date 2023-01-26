// RUN: %check_clang_tidy %s misc-use-anonymous-namespace %t -- -header-filter=.* -- -I%S/Inputs
#include "use-anonymous-namespace.h"

static void f1();
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: function 'f1' declared 'static', move to anonymous namespace instead [misc-use-anonymous-namespace]
static int v1;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'v1' declared 'static', move to anonymous namespace instead

namespace a {
  static void f3();
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: function 'f3' declared 'static', move to anonymous namespace instead
  static int v3;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: variable 'v3' declared 'static', move to anonymous namespace instead
}

// OK
void f5();
int v5;

// OK
namespace {
  void f6();
  int v6;
}

// OK
namespace a {
namespace {
  void f7();
  int v7;
}
}

// OK
struct Foo {
  static void f();
  static int x;
};

// OK
void foo()
{
  static int x;
}

// OK
static const int v8{123};
static constexpr int v9{123};
