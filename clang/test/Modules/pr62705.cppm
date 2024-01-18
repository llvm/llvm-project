// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 %t/a.cppm -std=c++20 -triple %itanium_abi_triple \
// RUN:     -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 %t/b.cppm -std=c++20 -triple %itanium_abi_triple \
// RUN:     -emit-module-interface -o %t/b.pcm \
// RUN:     -fmodule-file=a=%t/a.pcm
// RUN: %clang_cc1 %t/b.pcm -std=c++20 -triple %itanium_abi_triple \
// RUN:     -fmodule-file=a=%t/a.pcm -emit-llvm -o - | FileCheck %t/b.cppm

//--- foo.h
namespace n {

template<typename>
struct s0 {
	static int m;
};

template<typename T>
struct s1 {
	using type = s0<T>;
};

}

template<typename T>
void require(n::s1<T>) {
}

//--- a.cppm
module;

#include "foo.h"

export module a;

//--- b.cppm
module;

#include "foo.h"

export module b;
import a;

// Check the LLVM IR of module 'b' get generated correctly.
// CHECK: define{{.*}}@_ZGIW1b
