// RUN: %clang_cc1 -dwarf-version=4 -debug-info-kind=limited -emit-llvm < %s | FileCheck %s
// RUN: %clang_cc1 -dwarf-version=4 -debug-info-kind=line-tables-only -emit-llvm < %s | FileCheck --check-prefix=GMLT %s
// Two variables with same name in separate scope.
// Radar 8330217.
int main() {
	int j = 0;
	int k = 0;
// CHECK: !DILocalVariable(name: "i"
// CHECK-NEXT: !DILexicalBlock(

// GMLT-NOT: !DILexicalBlock
// GMLT: !DILexicalBlockFile({{.*}}, discriminator: 1)
// GMLT: !DILexicalBlockFile({{.*}}, discriminator: 2)
// Make sure we don't have any more lexical blocks because we don't need them in
// -gmlt.
// GMLT-NOT: !DILexicalBlock
	for (int i = 0; i < 10; i++)
		j++;
// CHECK: !DILocalVariable(name: "i"
// CHECK-NEXT: !DILexicalBlock(
// GMLT-NOT: !DILexicalBlock
	for (int i = 0; i < 10; i++)
		k++;
	return 0;
}
