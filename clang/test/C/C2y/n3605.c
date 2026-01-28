// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic %s
// expected-no-diagnostics
// RUN: %clang_cc1 -std=c2y %s -triple x86_64 --embed-dir=%S/Inputs -emit-llvm -o - | FileCheck %s

enum A{ a=(int)(_Generic(0, int: (2.5))) };
enum B{ b=(int)(_Generic(0, int: (2 + 1))) };

constexpr int c  = _Generic((float*)0, default: 0);

constexpr int d = _Generic((float*)0, default: (sizeof(c)));

char s[] = _Generic(0, default: ("word"));

// static_assert(1, _Generic(1, default: "Error Message"));

int value_of_a() {
	// CHECK: ret i32 2
	return a;
}

int value_of_b() {
	// CHECK: ret i32 3
	return b;
}

int value_of_c() {
	// CHECK: ret i32 0
	return c;
}

int value_of_d() {
	// CHECK: ret i32 4
	return d;
}

char *value_of_s() {
	// CHECK: ret ptr @s
    return s;
}

float value_of_float() {
	// CHECK: %f = alloca ptr, align 8
	// CHECK: store ptr null, ptr %f, align 8
	// CHECK: %0 = load ptr, ptr %f, align 8
	// CHECK: %call = call float %0()
	// CHECK: ret float %call

	float (*f)(void)  = _Generic(1, default: (void*)0);
	return f();
}