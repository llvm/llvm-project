// RUN: %clang_cc1 -g -emit-llvm < %s | FileCheck %s
// Two variables with same name in separate scope.
// Radar 8330217.
int main() {
	int j = 0;
	int k = 0;
// CHECK: DW_TAG_auto_variable
// CHECK-NEXT: DW_TAG_lexical_block
	for (int i = 0; i < 10; i++)
		j++;
	for (int i = 0; i < 10; i++)
		k++;
	return 0;
}
