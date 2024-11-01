// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -o %t/b.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/c.cppm -emit-module-interface -o %t/c.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/d.cppm -emit-module-interface -o %t/d.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/e.cpp -fsyntax-only -verify -fprebuilt-module-path=%t

//--- a.cppm
export module a;

struct WithCtor {
  WithCtor();
};

export template <typename T>
struct Getter {
  union {
    WithCtor container;
  };
};

//--- b.cppm
export module b;

import a;

export template <typename T>
class AnySpan {
 public:
  AnySpan();
  AnySpan(Getter<T> getter)
      : getter_(getter) {}

 private:
  Getter<T> getter_;
};

//--- c.cppm
export module c;
import b;

export inline void RegisterInt322(
   AnySpan<const int> sibling_field_nums) {
  sibling_field_nums = sibling_field_nums;
}

//--- d.cppm
// expected-no-diagnostics
export module d;
import c;
import b;

export inline void RegisterInt32(
   AnySpan<const int> sibling_field_nums = {}) {
  sibling_field_nums = sibling_field_nums;
}

//--- e.cpp
import d;
import b;

// expected-no-diagnostics
void foo(AnySpan<const int> s) {
  s = AnySpan<const int>(s);
}
