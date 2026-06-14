// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++26 -O3 -emit-llvm %s -o - | FileCheck %s
module;

struct A {};

struct B {
	int x;
	A   a;
	constexpr B(char *) { x = int(); }
	~B();
};

struct C {
	B b = "";
} inline c{};

export module foo;

// Just to make sure it won't crash
// CHECK: @_ZGIW3foo
