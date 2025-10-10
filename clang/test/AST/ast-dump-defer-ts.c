// Test without serialization:
// RUN: %clang_cc1 -std=c23 -fdefer-ts -ast-dump %s -triple x86_64-linux-gnu \
// RUN: | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -std=c23 -fdefer-ts -triple x86_64-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c23 -fdefer-ts -triple x86_64-linux-gnu -include-pch %t -ast-dump-all /dev/null \
// RUN: | FileCheck %s

static inline void f() {
  defer 3;
  defer { 4; }
  defer defer if (true) {}
}

// CHECK-LABEL: f 'void (void)' static inline
// CHECK-NEXT: `-CompoundStmt {{.*}} <col:24, line:14:1>
// CHECK-NEXT:   |-DeferStmt {{.*}} <line:11:3, col:9>
// CHECK-NEXT:   | `-IntegerLiteral {{.*}} <col:9> 'int' 3
// CHECK-NEXT:   |-DeferStmt {{.*}} <line:12:3, col:14>
// CHECK-NEXT:   | `-CompoundStmt {{.*}} <col:9, col:14>
// CHECK-NEXT:   |   `-IntegerLiteral {{.*}} <col:11> 'int' 4
// CHECK-NEXT:   `-DeferStmt {{.*}} <line:13:3, col:26>
// CHECK-NEXT:     `-DeferStmt {{.*}} <col:9, col:26>
// CHECK-NEXT:       `-IfStmt {{.*}} <col:15, col:26>
// CHECK-NEXT:         |-CXXBoolLiteralExpr {{.*}} <col:19> 'bool' true
// CHECK-NEXT:         `-CompoundStmt {{.*}} <col:25, col:26>
