// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,unix.Malloc,debug.ExprInspection -Wno-null-dereference -verify %s
// expected-no-diagnostics

#include "Inputs/system-header-simulator-cxx.h"

namespace Cxx11BraceInit {
  struct Foo {
    ~Foo() {}
  };

  void testInitializerList() {
    for (Foo foo : {Foo(), Foo()}) {}
  }
}

