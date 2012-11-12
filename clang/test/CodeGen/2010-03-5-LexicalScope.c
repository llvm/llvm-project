// RUN: %clang_cc1 -emit-llvm -O0 -g %s -o - | grep DW_TAG_lexical_block | count 3
int foo(int i) {
	if (i) {
		int j = 2;
	}
	else {
		int j = 3;
	}
	return i;
}
