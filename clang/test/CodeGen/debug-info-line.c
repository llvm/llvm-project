// RUN: %clang -emit-llvm -S -g %s -o - | FileCheck %s

// Radar 8396182
// There is only one lexical block, but we need a DILexicalBlock and two
// DILexicalBlockFile to correctly represent file info. This means we have
// two lexical blocks shown as the latter is also tagged as a lexical block.

int foo() {
  int i = 1;
# 4 "m.c"
# 1 "m.h" 1
  int j = 2;
# 2 "m.h"
# 5 "m.c" 2
  return i + j;
}

// CHECK: DW_TAG_lexical_block
// CHECK: DW_TAG_lexical_block
// CHECK: !"m.h"
// CHECK: DW_TAG_lexical_block
// CHECK: !"m.c"
// CHECK-NOT: DW_TAG_lexical_block
