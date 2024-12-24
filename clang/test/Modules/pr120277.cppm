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
// RUN: %clang_cc1 -std=c++20 %t/d.pcm -emit-llvm -o %t/d.ll \
// RUN:     -fprebuilt-module-path=%t
// RUN: cat %t/d.ll | FileCheck %t/d.cppm

//--- a.cppm
export module a;

export template<int>
struct a {
	static auto f() {
	}
};

//--- b.cppm
export module b;

import a;

void b() {
	a<0> t;
}

//--- c.cppm
export module c;

import a;

void c() {
	a<0>::f();
}

//--- d.cppm
export module d;

import a;
import b;
import c;

struct d {
	static void g() {
		a<0>::f();
		a<1>::f();
	}
};

// fine enough to check it won't crash
// CHECK: define
