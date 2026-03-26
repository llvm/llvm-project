// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s
// CHECK: !DILexicalBlock(
// CHECK: !DILexicalBlock(
int foo(int i) {
	if (i) {
		int j = 2;
	}
	else {
		int j = 3;
	}
	return i;
}
