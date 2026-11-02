// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

// A `#line 0` directive leaves the lexical block with no line number. Make sure
// we do not attach a column number in that case, since the IR verifier rejects
// a DILexicalBlock that has column info without line info.

void f(int x) {
#line 0
  if (x) {
    x++;
  }
}

// CHECK: distinct !DILexicalBlock(scope: !{{[0-9]+}}, file: !{{[0-9]+}})
// CHECK-NOT: !DILexicalBlock({{.*}}column:
