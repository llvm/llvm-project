// From https://github.com/llvm/llvm-project/issues/221548
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++23 -emit-module-interface %t/stdish.cppm -o %t/stdish.pcm
// RUN: %clang_cc1 -std=c++23 -emit-module-interface %t/wrap.cppm -o %t/wrap.pcm
// RUN: %clang_cc1 -std=c++23 -fmodule-file=stdish=%t/stdish.pcm -fmodule-file=wrap=%t/wrap.pcm -emit-module-interface %t/side.cppm -o %t/side.pcm
// RUN: %clang_cc1 -std=c++23 -fmodule-file=stdish=%t/stdish.pcm -fmodule-file=wrap=%t/wrap.pcm -fmodule-file=side=%t/side.pcm %t/use.cc -fsyntax-only -verify
//
// Test again with reduced BMI.
// RUN: %clang_cc1 -std=c++23 -emit-reduced-module-interface %t/stdish.cppm -o %t/stdish.pcm
// RUN: %clang_cc1 -std=c++23 -emit-reduced-module-interface %t/wrap.cppm -o %t/wrap.pcm
// RUN: %clang_cc1 -std=c++23 -fmodule-file=stdish=%t/stdish.pcm -fmodule-file=wrap=%t/wrap.pcm -emit-reduced-module-interface %t/side.cppm -o %t/side.pcm
// RUN: %clang_cc1 -std=c++23 -fmodule-file=stdish=%t/stdish.pcm -fmodule-file=wrap=%t/wrap.pcm -fmodule-file=side=%t/side.pcm %t/use.cc -fsyntax-only -verify

//--- mine.h
#pragma once
namespace mine {

template <class value_type>
struct box {
  value_type value{};
  template <class other_kind>
  constexpr value_type twice(other_kind extra) const;
};

template <class value_type>
template <class other_kind>
constexpr value_type box<value_type>::twice(other_kind extra) const {
  value_type held = value + static_cast<value_type>(extra);
  auto go = [&] { held = held + held; };
  go();
  return held;
}

}  // namespace mine

//--- stdish.cppm
module;
#include "mine.h"
export module stdish;
export namespace mine { using mine::box; }

//--- wrap.cppm
module;
#include "mine.h"
export module wrap;
export template <class value_type> using holder = mine::box<value_type>;

//--- side.cppm
export module side;
import stdish;
export constexpr int one() { return mine::box<long>{21}.twice(0) == 42 ? 42 : 0; }

//--- use.cc
// expected-no-diagnostics
import wrap;
import stdish;
import side;

int main() {
  holder<int> spare{1};
  static_assert(one() == 42);
  static_assert(mine::box<int>{21}.twice(0) == 42);
}
