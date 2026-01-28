// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/flyweight.cppm -emit-reduced-module-interface -o %t/flyweight.pcm
// RUN: %clang_cc1 -std=c++20 %t/account.cppm -emit-reduced-module-interface -o %t/account.pcm -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/core.cppm -emit-reduced-module-interface -o %t/core.pcm -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/core.cppm -fprebuilt-module-path=%t -emit-llvm -disable-llvm-passes -o - | FileCheck %t/core.cppm

//--- flyweight.cppm
module;
template <typename> struct flyweight_core {
  static bool init() { (void)__builtin_operator_new(2); return true; }
  static bool static_initializer;
};
template <typename T> bool flyweight_core<T>::static_initializer = init();
export module flyweight;
export template <class> void flyweight() {
  (void)flyweight_core<int>::static_initializer;
}

//--- account.cppm
export module account;
import flyweight;
export void account() {
  (void)::flyweight<int>;
}

//--- core.cppm
export module core;
import account;

extern "C" void core() {}

// Fine enough to check it won't crash.
// CHECK-NOT: init
// CHECK-NOT: static_initializer
// CHECK: define {{.*}}@core(
