// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -verify -w -std=c++20 -fmodule-name=h1.h -emit-header-unit -xc++-user-header h1.h -o h1.pcm
// RUN: %clang_cc1 -verify -w -std=c++20 -fmodule-map-file=module.modulemap -fmodule-file=h1.h=h1.pcm main.cpp -o main.o

//--- module.modulemap
module "h1.h" {
  header "h1.h"
  export *
}

//--- h0.h
// expected-no-diagnostics
#pragma once

template <typename T> struct A {
  union {
    struct {
      T x, y, z;
    };
  };
  constexpr A(T, T, T) : x(), y(), z() {}
};
typedef A<float> packed_vec3;

//--- h1.h
// expected-no-diagnostics
#pragma once

#include "h0.h"

constexpr packed_vec3 kMessThingsUp = packed_vec3(5.0f, 5.0f, 5.0f);

//--- main.cpp
// expected-no-diagnostics
#include "h0.h"

static_assert(sizeof(packed_vec3) == sizeof(float) * 3);
static_assert(alignof(packed_vec3) == sizeof(float));

import "h1.h";

constexpr packed_vec3 kDefaultHalfExtents = packed_vec3(5.0f, 5.0f, 5.0f);
