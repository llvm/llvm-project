// Test that functions excluded from explicit instantiation, such as libc++'s
// hidden ones, are emitted by every user of a module built with
// -fmodules-codegen instead of only in the module's object file.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -triple x86_64-linux-gnu -x c++ -fmodules-codegen -fmodules -emit-module -fmodule-name=foo %t/foo.modulemap -o %t/foo.pcm

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %t/foo.pcm -o - | FileCheck %s --check-prefix=DEF
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fmodules -fmodule-file=%t/foo.pcm -I%t -disable-llvm-passes -emit-llvm %t/test.cpp -o - | FileCheck %s --check-prefix=USE

// DEF-DAG: define weak_odr void @_Z11instantiatev()
// DEF-DAG: define weak_odr void @_ZN1SIiE5homedEv(
// DEF-DAG: define linkonce_odr hidden void @_ZN1SIiE8excludedEv(

// USE-DAG: declare void @_ZN1SIiE5homedEv(
// USE-DAG: define linkonce_odr hidden void @_ZN1SIiE8excludedEv(

//--- foo.h
template <typename T> struct S {
  void homed() {}
  __attribute__((visibility("hidden"), exclude_from_explicit_instantiation))
  void excluded() {}
};

inline void instantiate() {
  S<int> s;
  s.homed();
  s.excluded();
}

//--- test.cpp
#include "foo.h"

void test() {
  S<int> s;
  s.homed();
  s.excluded();
}

//--- foo.modulemap
module foo { header "foo.h" }
