// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// CHECK: FTrivial
// CHECK: DefinitionData
// CHECK-SAME: pass_in_registers

// CHECK: FNonTrivial
// CHECK-NOT: pass_in_registers
// CHECK: DefaultConstructor

void expr() {
  FTrivial f1;
  FNonTrivial f2;
}
