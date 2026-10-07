// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -emit-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t \
// RUN:   -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -emit-obj -fmodules-reduced-bmi \
// RUN:   -fmodule-output=%t/M.pcm -o %t/M.o
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t \
// RUN:   -fsyntax-only -verify

//--- support.h
using size_t = decltype(sizeof(0));

struct placement_tag {};
inline void *operator new(size_t, void *p) { return p; }
inline void *operator new(size_t, void *p, placement_tag) { return p; }

namespace ranges {
struct reverse_fn {};
inline constexpr reverse_fn reverse;

template <class Range>
bool operator|(Range &&, reverse_fn) {
  return true;
}
} // namespace ranges

//--- M.cppm
module;
#include "support.h"
export module M;

export template <class T>
struct box {
  alignas(T) unsigned char storage[sizeof(T)];

  void construct(T value) { ::new (static_cast<void *>(storage)) T(value); }

  void construct_tagged(T value) {
    ::new (static_cast<void *>(storage), placement_tag{}) T(value);
  }

  bool reverse() { return *this | ranges::reverse; }
};

//--- use.cpp
// expected-no-diagnostics
import M;

void use() {
  box<int> b;
  b.construct(42);
  b.construct_tagged(43);
  (void)b.reverse();
}
