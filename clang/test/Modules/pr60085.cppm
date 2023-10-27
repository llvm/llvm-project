// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/d.cppm \
// RUN:     -emit-module-interface -o %t/d.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/c.cppm \
// RUN:     -emit-module-interface -o %t/c.pcm -fmodule-file=%t/d.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/b.cppm \
// RUN:     -emit-module-interface -o %t/b.pcm -fmodule-file=%t/d.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/a.cppm \
// RUN:     -emit-module-interface -o %t/a.pcm -fmodule-file=%t/d.pcm \
// RUN:     -fmodule-file=%t/c.pcm -fmodule-file=%t/b.pcm 
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/a.pcm \
// RUN:     -S -emit-llvm -disable-llvm-passes -o - | FileCheck %t/a.cppm
//
// Use -fmodule-file=<module-name>=<BMI-path>
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/d.cppm \
// RUN:     -emit-module-interface -o %t/d.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/c.cppm \
// RUN:     -emit-module-interface -o %t/c.pcm -fmodule-file=%t/d.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/b.cppm \
// RUN:     -emit-module-interface -o %t/b.pcm -fmodule-file=%t/d.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/a.cppm \
// RUN:     -emit-module-interface -o %t/a.pcm -fmodule-file=%t/d.pcm \
// RUN:     -fmodule-file=%t/c.pcm -fmodule-file=%t/b.pcm 
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/a.pcm \
// RUN:     -S -emit-llvm -disable-llvm-passes -o - | FileCheck %t/a.cppm

//--- d.cppm
export module d;

export template<typename>
struct integer {
	using type = int;
	
	static constexpr auto value() {
		return 0;
	}
	
	friend constexpr void f(integer const x) {
		x.value();
	}
};

export constexpr void ddd(auto const value) {
	f(value);
}


template<typename T>
constexpr auto dd = T();

export template<typename T>
constexpr auto d() {
	dd<T>;
}

//--- c.cppm
export module c;

import d;

template<typename T>
auto cc = T();

auto c() {
	cc<integer<int>>;
	integer<int>().value();
}

//--- b.cppm
export module b;

import d;

auto b() {
	integer<int>::type;
}

//--- a.cppm
export module a;

import b;
import c;
import d;

constexpr void aa() {
	d<integer<unsigned>>();
	ddd(integer<int>());
}

export extern "C" void a() {
	aa();
}

// Checks that we emit the IR successfully.
// CHECK: define{{.*}}@a(
