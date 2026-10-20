// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/module1.cppm -emit-module-interface -o %t/module1.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-file=module1=%t/module1.pcm  %t/module2.cppm \
// RUN:     -emit-module-interface -o %t/module2.pcm
// RUN: %clang_cc1 -std=c++20 %t/module2.pcm -fmodule-file=module1=%t/module1.pcm \
// RUN:     -emit-llvm -o - | FileCheck %t/module2.cppm

//--- test.h
template<typename T>
struct Test {
  template<typename U>
  friend class Test;
};

//--- module1.cppm
module;
#include "test.h"
export module module1;
export void f1(Test<int>) {}

//--- module2.cppm
module;
#include "test.h"
export module module2;
import module1;
export void f2(Test<float>) {}

extern "C" void func() {}

// Fine enough to check the IR is emitted correctly.
// CHECK: define{{.*}}@func
