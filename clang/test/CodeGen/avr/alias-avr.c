// RUN: %clang_cc1 -triple avr-unknown-unknown -emit-llvm -o - %s | FileCheck %s

int mul(int a, int b) {
	return a * b;
}

// CHECK: @multiply ={{.*}} alias i16 (i16, i16), ptr addrspace(1) @mul
int multiply(int x, int y) __attribute__((alias("mul")));

// Make sure the correct address space is used when creating an alias that needs
// a pointer cast.
// CHECK: @smallmul = alias i8 (i16, i16), ptr addrspace(1) @mul
char smallmul(int a, int b) __attribute__((alias("mul")));
