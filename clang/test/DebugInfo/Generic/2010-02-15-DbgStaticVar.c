// RUN: %clang_cc1 -debug-info-kind=limited -emit-llvm %s -o - | FileCheck %s
// Test to check intentionally empty linkage name for a static variable.
static int foo(int a)
{
	static int b = 1;
	return b+a;
}

int main(void) {
	int j = foo(1);
	return 0;
}
// CHECK: !DIGlobalVariable(name: "b",
// CHECK-NOT:               linkageName:
// CHECK-SAME:              ){{$}}
