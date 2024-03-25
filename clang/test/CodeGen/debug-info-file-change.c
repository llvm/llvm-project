// RUN: %clang -emit-llvm -S -g %s -o - | FileCheck %s

// There are no lexical blocks, but we need two DILexicalBlockFiles to
// correctly represent file info.

int foo(void) {
  int i = 1;
# 4 "m.c"
# 1 "m.h" 1
  int j = 2;
# 2 "m.h"
# 5 "m.c" 2
  return i + j;
}

// CHECK-NOT: !DILexicalBlock
// CHECK: !DILexicalBlockFile({{.*}}file: ![[MH:[0-9]+]]
// CHECK: !DIFile(filename: "m.h"
// CHECK: !DILexicalBlockFile({{.*}}file: ![[MC:[0-9]+]]
// CHECK: !DIFile(filename: "m.c"
// CHECK-NOT: !DILexicalBlock
