// From https://github.com/llvm/llvm-project/issues/61065
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -o %t/b.pcm \
// RUN:     -fprebuilt-module-path=%t
// DISABLED: %clang_cc1 -std=c++20 %t/c.cppm -emit-module-interface -o %t/c.pcm \
// DISABLED:     -fprebuilt-module-path=%t
// DISABLED: %clang_cc1 -std=c++20 %t/d.cpp -fsyntax-only -verify -fprebuilt-module-path=%t

//--- a.cppm
export module a;

struct base {
	base(int) {}
};

export struct a : base {
	using base::base;
};

//--- b.cppm
export module b;

import a;

a b() {
	return a(1);
}

//--- c.cppm
export module c;

import a;
import b;

struct noncopyable {
	noncopyable(noncopyable const &) = delete;
    noncopyable() = default;
};

export struct c {
	noncopyable c0;
	a c1 = 43;
    c() = default;
};

//--- d.cpp
// expected-no-diagnostics
import c;
void d() {
    c _;
}
