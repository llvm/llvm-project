// RUN: %clang_cc1 -xc++ -std=c++23 -ast-dump %s | FileCheck %s

int inline consteval operator""_u32(unsigned long long val) {
  return val;
}

void udl() {
  (void)(0_u32 + 1_u32);
}

// CHECK: `-BinaryOperator {{.+}} <col:10, col:18> 'int' '+'
// CHECK-NEXT: |-ConstantExpr {{.+}} <col:10> 'int'
// CHECK-NEXT: | |-value: Int 0
// CHECK-NEXT: | `-UserDefinedLiteral {{.+}} <col:10> 'int'
// CHECK: `-ConstantExpr {{.+}} <col:18> 'int'
// CHECK-NEXT:   |-value: Int 1
// CHECK-NEXT:   `-UserDefinedLiteral {{.+}} <col:18> 'int'
