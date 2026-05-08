// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic %s
// RUN: %clang_cc1 -verify -std=c23 -Wall -pedantic %s
// RUN: %clang_cc1 -verify -std=c17 -Wall -pedantic %s
// RUN: %clang_cc1 -verify -std=c11 -Wall -pedantic %s
// expected-no-diagnostics
// RUN: %clang_cc1 -std=c2y %s -triple x86_64 --embed-dir=%S/Inputs -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -std=c23 %s -triple x86_64 --embed-dir=%S/Inputs -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -std=c17 %s -triple x86_64 --embed-dir=%S/Inputs -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -std=c11 %s -triple x86_64 --embed-dir=%S/Inputs -emit-llvm -o - | FileCheck %s

enum A{ a=(int)(_Generic(0, int: (2.5))) };
enum B{ b=(int)(_Generic(0, int: (2 + 1))) };

int c = _Generic((float*)0, default: 0);
int d = _Generic((float*)0, default: (sizeof(c)));

char s[] = _Generic(0, default: ("word"));

int value_of_a(void) {
	// CHECK: ret i32 2
	return a;
}

int value_of_b(void) {
	// CHECK: ret i32 3
	return b;
}

int value_of_c(void) {
	// CHECK: %0 = load i32, ptr @c, align 4
	// CHECK: ret i32 %0
	return c;
}

int value_of_d(void) {
	// CHECK: %0 = load i32, ptr @d, align 4
	// CHECK: ret i32 %0
	return d;
}

char *value_of_s(void) {
	// CHECK: ret ptr @s
    return s;
}

float value_of_float(void) {
	// CHECK: %f = alloca ptr, align 8
	// CHECK: store ptr null, ptr %f, align 8
	// CHECK: %0 = load ptr, ptr %f, align 8
	// CHECK: %call = call float %0()
	// CHECK: ret float %call

	float (*f)(void)  = _Generic(1, default: (void*)0);
	return f();
}


