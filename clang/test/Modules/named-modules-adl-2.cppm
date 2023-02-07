// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -fmodule-file=%t/a.pcm -emit-module-interface -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 %t/c.cppm -fmodule-file=%t/b.pcm -fsyntax-only -verify

//--- a.cppm
export module a;

export template<typename T>
void a(T x) {
	+x;
}

//--- b.h
struct s {
};
void operator+(s) {
}

//--- b.cppm
module;
#include "b.h"
export module b;
import a;

export template<typename T>
void b() {
	a(s());
}

//--- c.cppm
// expected-no-diagnostics
export module c;
import b;

void c() {
	b<int>();
}
