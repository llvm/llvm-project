// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -fmodule-file=a=%t/a.pcm -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -fmodule-file=a=%t/a.pcm -fsyntax-only -verify

//--- a.h
namespace n {

struct s { };

void operator+(s, int) {
}

} // namespace n

//--- a.cppm
module;
#include "a.h"
export module a;

export template<typename T>
void a(T x) {
	n::s() + x;
}

//--- b.cppm
// expected-no-diagnostics
export module b;
import a;

void b() {
	a(0);
}
