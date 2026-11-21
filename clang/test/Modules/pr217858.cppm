// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++23 -triple %itanium_abi_triple \
// RUN:   -emit-module-interface %t/lib.cppm -o %t/Lib.pcm
// RUN: %clang_cc1 -std=c++23 -triple %itanium_abi_triple \
// RUN:   -emit-module-interface %t/mod.cppm -o %t/Mod.pcm \
// RUN:   -fmodule-file=Lib=%t/Lib.pcm
// RUN: %clang_cc1 -std=c++23 -triple %itanium_abi_triple -emit-obj \
// RUN:   %t/main.cpp -o %t/main.o -fmodule-file=Lib=%t/Lib.pcm \
// RUN:   -fmodule-file=Mod=%t/Mod.pcm

// Test again with reduced BMI.
// RUN: %clang_cc1 -std=c++23 -triple %itanium_abi_triple \
// RUN:   -emit-reduced-module-interface %t/lib.cppm -o %t/Lib.pcm
// RUN: %clang_cc1 -std=c++23 -triple %itanium_abi_triple \
// RUN:   -emit-reduced-module-interface %t/mod.cppm -o %t/Mod.pcm \
// RUN:   -fmodule-file=Lib=%t/Lib.pcm
// RUN: %clang_cc1 -std=c++23 -triple %itanium_abi_triple -emit-obj \
// RUN:   %t/main.cpp -o %t/main.o -fmodule-file=Lib=%t/Lib.pcm \
// RUN:   -fmodule-file=Mod=%t/Mod.pcm

//--- lib.cppm
export module Lib;

export template <class Callback> auto make_closure(Callback &callback) {
  return [&callback](auto &arg) noexcept(noexcept(callback(arg))) {
    callback(arg);
  };
}

export template <class Callback, class Arg>
void for_each(Callback &&callback, Arg &arg)
    noexcept(noexcept(make_closure(callback)(arg))) {}

export template <class It> struct iterator {
  It current;
  void operator++()
      noexcept(noexcept(for_each([](auto &it) { ++it; }, current))) {}
};

//--- mod.cppm
export module Mod;
import Lib;

export inline int test() {
  int data[4]{};
  iterator<int *> it{data};
  ++it;
  return static_cast<int>(it.current - data);
}

//--- main.cpp
import Mod;
int main() { return test(); }
