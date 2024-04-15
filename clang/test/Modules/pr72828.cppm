// Test that we can handle capturing structured bindings.
//
// RUN: rm -fr %t
// RUN: mkdir %t
//
// RUN: %clang_cc1 -std=c++23 -triple %itanium_abi_triple \
// RUN:     %s -emit-module-interface -o %t/m.pcm
// RUN: %clang_cc1 -std=c++23 -triple %itanium_abi_triple \
// RUN:     -S -emit-llvm -disable-llvm-passes %t/m.pcm \
// RUN:     -o - | FileCheck %s

export module m;

struct s {
	int m;
};

void f() {
	auto [x] = s();
	(void) [x] {};
}

// Check that we can generate the LLVM IR expectedly.
// CHECK: define{{.*}}@_ZGIW1m
