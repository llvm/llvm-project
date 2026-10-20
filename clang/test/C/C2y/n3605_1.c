// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic %s
// RUN: %clang_cc1 -verify -std=c23 -Wall -pedantic %s
// expected-no-diagnostics
// RUN: %clang_cc1 -std=c2y %s -triple x86_64 --embed-dir=%S/Inputs -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -std=c23 %s -triple x86_64 --embed-dir=%S/Inputs -emit-llvm -o - | FileCheck %s

constexpr int a = _Generic((float*)0, default: 0);
constexpr int b = _Generic((float*)0, default: (sizeof(a)));

int value_of_a(void) {
	// CHECK: ret i32 0
	return a;
}

int value_of_b(void) {
	// CHECK: ret i32 4
	return b;
}

