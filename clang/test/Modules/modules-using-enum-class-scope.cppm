// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 -fmodules -fimplicit-module-maps -x c++ -I%t \
// RUN:   -emit-module -fmodule-name=ModuleA -o %t/ModuleA.pcm \
// RUN:   %t/module.modulemap
//
// RUN: %clang_cc1 -std=c++20 -fmodules -fimplicit-module-maps -x c++ -I%t \
// RUN:   -emit-module -fmodule-name=ModuleB -o %t/ModuleB.pcm \
// RUN:   %t/module.modulemap
//
// RUN: %clang_cc1 -std=c++20 -fmodules -fimplicit-module-maps -x c++ -I%t \
// RUN:   -fmodule-file=%t/ModuleA.pcm \
// RUN:   -fmodule-file=%t/ModuleB.pcm \
// RUN:   -verify %t/main.cpp

//--- common.h
#pragma once
struct MyStruct {
  enum class MyEnum { A };
  using enum MyEnum;
};

//--- a.h
#pragma once
#include "common.h"

//--- b.h
#pragma once
#include "common.h"

//--- module.modulemap
module ModuleA {
  header "a.h"
  export *
}

module ModuleB {
  header "b.h"
  export *
}

//--- main.cpp
#include "a.h"
#include "b.h"

inline void use() {
  auto x = MyStruct::A;
}

// expected-no-diagnostics
