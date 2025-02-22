// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -verify -std=c++20 -Werror=uninitialized -xc++ -emit-module module.cppmap -fmodule-name=mock_resolver -o mock_resolver.pcm
// RUN: %clang_cc1 -verify -std=c++20 -Werror=uninitialized -xc++ -emit-module module.cppmap -fmodule-name=sql_internal -o sql_internal.pcm
// RUN: %clang_cc1 -verify -std=c++20 -Werror=uninitialized -xc++ -fmodule-file=mock_resolver.pcm -fmodule-file=sql_internal.pcm main.cc -o main.o

//--- module.cppmap
module "mock_resolver" {
  export *
  module "mock_resolver.h" {
    export *
    header "mock_resolver.h"
  }
}

module "sql_internal" {
  export *
  module "sql_transform_builder.h" {
    export *
    header "sql_transform_builder.h"
  }
}

//--- set_bits2.h
// expected-no-diagnostics
#pragma once

template <typename T>
void fwd(const T& x) {}

namespace vox::bitset {

template <typename TFunc>
void ForEachSetBit2(const TFunc&) {
  fwd([](int) {
    const int bit_index_base = 0;
    (void)[&](int) {
      int v = bit_index_base;
    };
  });
}

}  // namespace vox::bitset

//--- sql_transform_builder.h
// expected-no-diagnostics
#pragma once

#include "set_bits2.h"

class QualifyingSet3 {
 public:
  void GetIndexes() const {
    vox::bitset::ForEachSetBit2([]() {});
  }
};

template <typename T>
void DoTransform() {
  vox::bitset::ForEachSetBit2([]() {});
}

//--- mock_resolver.h
// expected-no-diagnostics
#pragma once 
#include "set_bits2.h"

class QualifyingSet2 {
 public:
  void GetIndexes() const {
    vox::bitset::ForEachSetBit2([]() {});
  }
};

//--- main.cc
// expected-no-diagnostics
#include "sql_transform_builder.h"

template <typename Callable>
void get(const Callable& fn) {
  fwd<Callable>(fn);
}

namespace {

void test() {
  get([]() {});
  DoTransform<int>();
}

} // namespace
