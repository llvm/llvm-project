// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %t/main.cpp -o %t/main.o

//--- V.h
#ifndef V_H
#define V_H

class A {
public:
  constexpr A() { }
  constexpr ~A() { }
};

template <typename T>
class V {
public:
  V() = default;

  constexpr V(int n, const A& a = A()) {}
};

#endif

//--- inst1.h
#include "V.h"

static void inst1() {
  V<int> v;
}

//--- inst2.h
#include "V.h"

static void inst2() {
  V<int> v(100);
}

//--- module.modulemap
module "M" {
  export *
  module "V.h" {
    export *
    header "V.h"
  }
  module "inst1.h" {
    export *
    header "inst1.h"
  }
}

module "inst2.h" {
  export *
  header "inst2.h"
}

//--- main.cpp
#include "V.h"
#include "inst2.h"

static void m() {
  static V<int> v(100);
}
