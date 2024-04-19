// https://github.com/llvm/llvm-project/issues/60890
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/b.cppm -fprebuilt-module-path=%t -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/c.cppm -fprebuilt-module-path=%t -o %t/c.pcm
// RUN: %clang_cc1 -std=c++20 %t/d.cpp -fprebuilt-module-path=%t -S -emit-llvm -o -

// Test again with reduced BMI
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/b.cppm -fprebuilt-module-path=%t -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/c.cppm -fprebuilt-module-path=%t -o %t/c.pcm
// RUN: %clang_cc1 -std=c++20 %t/d.cpp -fprebuilt-module-path=%t -S -emit-llvm -o -

//--- a.cppm
export module a;
 
export template<typename T>
struct a {
	friend void aa(a x) requires(true) {}
	void aaa() requires(true) {}
};

export template struct a<double>;

export template<typename T>
void foo(T) requires(true) {}

export template void foo<double>(double);

export template <typename T>
class A {
	friend void foo<>(A);
};

//--- b.cppm
export module b;

import a;

void b() {
    a<int> _;
	a<double> __;
}

//--- c.cppm
export module c;

import a;

struct c {
	void f() const {
		a<int> _;
		aa(_);
		_.aaa();

		a<double> __;
		aa(__);
		__.aaa();

		foo<int>(5);
		foo<double>(3.0);
		foo(A<int>());
	}
};

//--- d.cpp
// expected-no-diagnostics
import a;
import b;
import c;

void d() {
	a<int> _;
	aa(_);
	_.aaa();

	a<double> __;
	aa(__);
	__.aaa();

	foo<int>(5);
	foo<double>(3.0);
	foo(A<int>());
}
