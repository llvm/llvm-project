// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -o %t/b.pcm \
// RUN:   -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/c.cppm -emit-module-interface -o %t/c.pcm \
// RUN:   -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/d.cpp -fsyntax-only -verify \
// RUN:   -fprebuilt-module-path=%t

//--- a.cppm
export module a;

template<typename>
constexpr auto impl = true;

export template<typename T>
void a() {
}

export template<typename T> requires impl<T>
void a() {
}

//--- b.cppm
export module b;

import a;

static void b() {
	a<void>();
}

//--- c.cppm
export module c;

import a;

static void c() {
	a<void>();
}

//--- d.cpp
// expected-no-diagnostics
import a;
import b;
import c;

static void d() {
	a<void>();
}
