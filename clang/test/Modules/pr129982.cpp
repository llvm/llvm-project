// If this test fails, it should be investigated under Debug builds.
// Before the PR, this test was violating an assertion.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-obj -fmodules \
// RUN:  -fmodule-map-file=%t/module.modulemap \
// RUN:  -fmodules-cache-path=%t %t/a.cpp

//--- module.modulemap
module ebo {
  header "ebo.h"
}

module fwd {
  header "fwd.h"
}

module s {
  header "s.h"
  export *
}

module mod {
  header "a.h"
  header "b.h"
}

//--- ebo.h
#pragma once

namespace N { inline namespace __1 {

template <typename T>
struct EBO : T {
  EBO() = default;
};

}}

//--- fwd.h
#pragma once

namespace N { inline namespace __1 {

template <typename T>
struct Empty;

template <typename T>
struct BS;

using S = BS<Empty<char>>;

}}

//--- s.h
#pragma once

#include "fwd.h"
#include "ebo.h"

namespace N { inline namespace __1 {

template <typename T>
struct Empty {};

template <typename T>
struct BS {
    EBO<T> _;
    void f();
};

extern template void BS<Empty<char>>::f();

}}

//--- b.h
#pragma once

#include "s.h"

struct B {
  void f() {
    N::S{}.f();
  }
};

//--- a.h
#pragma once

#include "s.h"

struct A {
  void f(int) {}
  void f(const N::S &) {}

  void g();
};

//--- a.cpp
#include "a.h"

void A::g() { f(0); }

// expected-no-diagnostics
