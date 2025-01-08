// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -o %t/b.pcm \
// RUN:   -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/test.cpp -fsyntax-only -verify \
// RUN:   -fprebuilt-module-path=%t

//--- a.cppm
export module a;

namespace n {
template<typename, int...>
struct integer_sequence {

};

export template<typename>
using make_integer_sequence = __make_integer_seq<integer_sequence, int, 0>;
}

//--- b.cppm
export module b;
import a;

export template<typename T>
void b() {
	n::make_integer_sequence<T>();
}

//--- test.cpp
// expected-no-diagnostics
import b;
void test() {
  b<int>();
}
