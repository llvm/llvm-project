// RUN: %clang_cc1 -triple avr-unknown-unknown -emit-llvm -o - %s | FileCheck %s

int mul(int a, int b) {
	return a * b;
}

// CHECK: @multiply = alias i16 (i16, i16), i16 (i16, i16) addrspace(1)* @mul
int multiply(int x, int y) __attribute__((alias("mul")));
