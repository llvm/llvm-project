// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// RUN: clang-import-test -dump-ast -x objective-c++ -import %S/Inputs/F.m -expression %s | FileCheck %s

// CHECK: ObjCAutoreleasePoolStmt
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt

void expr() {
  f();
}
