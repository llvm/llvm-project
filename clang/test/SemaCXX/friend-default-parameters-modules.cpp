// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -std=c++20 -fmodule-map-file=modules.map -xc++ -emit-module -fmodule-name=foo modules.map -o foo.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-map-file=modules.map -O1 -emit-obj main.cc -verify -fmodule-file=foo.pcm

//--- modules.map
module "foo" {
  export *
  module "foo.h" {
    export *
    header "foo.h"
  }
}

//--- foo.h
#pragma once

template <int>
void Create(const void* = nullptr);

template <int>
struct ObjImpl {
  template <int>
  friend void ::Create(const void*);
};

template <int I>
void Create(const void*) {
  (void) ObjImpl<I>{};
}

//--- main.cc
// expected-no-diagnostics
#include "foo.h"

int main() {
  Create<42>();
}
