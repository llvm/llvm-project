// Test case from https://github.com/llvm/llvm-project/issues/60036
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -fprebuilt-module-path=%t -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 %t/c.cppm -emit-module-interface -fprebuilt-module-path=%t -o %t/c.pcm
// RUN: %clang_cc1 -std=c++20 %t/d.cppm -emit-module-interface -fprebuilt-module-path=%t -o %t/d.pcm
// RUN: %clang_cc1 -std=c++20 %t/e.cppm -emit-module-interface -fprebuilt-module-path=%t -o %t/e.pcm
// RUN: %clang_cc1 -std=c++20 %t/f.cppm -emit-module-interface -fprebuilt-module-path=%t -o %t/f.pcm
// RUN: %clang_cc1 -std=c++20 %t/g.cppm -fprebuilt-module-path=%t -verify -fsyntax-only 
//
// Tests that the behavior is fine with specifying module file with `-fmodule-file`.
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -fmodule-file=a=%t/a.pcm -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 %t/c.cppm -emit-module-interface -fmodule-file=a=%t/a.pcm -o %t/c.pcm
// RUN: %clang_cc1 -std=c++20 %t/d.cppm -emit-module-interface  -o %t/d.pcm
// RUN: %clang_cc1 -std=c++20 %t/e.cppm -emit-module-interface -fmodule-file=a=%t/a.pcm -fmodule-file=b=%t/b.pcm \
// RUN:		-fmodule-file=c=%t/c.pcm -fmodule-file=d=%t/d.pcm -o %t/e.pcm
// RUN: %clang_cc1 -std=c++20 %t/f.cppm -emit-module-interface -fmodule-file=a=%t/a.pcm -fmodule-file=b=%t/b.pcm -fmodule-file=c=%t/c.pcm -fmodule-file=d=%t/d.pcm -o %t/f.pcm
// RUN: %clang_cc1 -std=c++20 %t/g.cppm -fmodule-file=a=%t/a.pcm -fmodule-file=b=%t/b.pcm \
// RUN:		-fmodule-file=c=%t/c.pcm -fmodule-file=d=%t/d.pcm -fmodule-file=e=%t/e.pcm \
// RUN:		-fmodule-file=f=%t/f.pcm -verify -fsyntax-only 

// Test again with reduced BMI
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-reduced-module-interface -fprebuilt-module-path=%t -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 %t/c.cppm -emit-reduced-module-interface -fprebuilt-module-path=%t -o %t/c.pcm
// RUN: %clang_cc1 -std=c++20 %t/d.cppm -emit-reduced-module-interface -fprebuilt-module-path=%t -o %t/d.pcm
// RUN: %clang_cc1 -std=c++20 %t/e.cppm -emit-reduced-module-interface -fprebuilt-module-path=%t -o %t/e.pcm
// RUN: %clang_cc1 -std=c++20 %t/f.cppm -emit-reduced-module-interface -fprebuilt-module-path=%t -o %t/f.pcm
// RUN: %clang_cc1 -std=c++20 %t/g.cppm -fprebuilt-module-path=%t -verify -fsyntax-only 


//--- a.cppm
export module a;

export template<typename>
struct a;

template<typename T>
struct a<T &> {
	using type = char;
};

//--- b.cppm
export module b;

import a;

typename a<char &>::type;

//--- c.cppm
export module c;

import a;

typename a<char &>::type;

//--- d.cppm
export module d;

export template<typename>
struct d {
	d() {}
	explicit d(int) requires(true) {}
	d(auto &&) {}
};

//--- e.cppm
export module e;

import d;

auto r = d<int>();

//--- f.cppm
export module f;

import a;
import b;
import c;
import d;

template<typename T>
struct array {
	friend void fr(array<T> const &) {
	}
};

array() -> array<typename a<char &>::type>;

struct wrap {
	d<int> m;
};

template<typename T>
int f1(T) {
	return 1;
}

void f2() {
	d<int> view;
	int x = f1(view);
	typename a<decltype([x]{}) &>::type;
	wrap w;
}

//--- g.cppm
// expected-no-diagnostics
export module g;

import e;
import f;
