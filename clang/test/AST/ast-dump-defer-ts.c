// Test without serialization:
// RUN: %clang_cc1 -std=c23 -fdefer-ts -ast-dump %s -triple x86_64-linux-gnu \
// RUN: | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -std=c23 -fdefer-ts -triple x86_64-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c23 -fdefer-ts -triple x86_64-linux-gnu -include-pch %t -ast-dump-all /dev/null \
// RUN: | FileCheck %s

static inline void f() {
  _Defer 3;
  _Defer { 4; }
  _Defer _Defer if (true) {}
}

// CHECK-LABEL: f 'void (void)' static inline
// CHECK-NEXT:  `-CompoundStmt {{.*}} <col:24, line:14:1>
// CHECK-NEXT:    |-DeferStmt {{.*}} <line:11:3, col:10>
// CHECK-NEXT:    | `-IntegerLiteral {{.*}} <col:10> 'int' 3
// CHECK-NEXT:    |-DeferStmt {{.*}} <line:12:3, col:15>
// CHECK-NEXT:    | `-CompoundStmt {{.*}} <col:10, col:15>
// CHECK-NEXT:    |   `-IntegerLiteral {{.*}} <col:12> 'int' 4
// CHECK-NEXT:    `-DeferStmt {{.*}} <line:13:3, col:28>
// CHECK-NEXT:      `-DeferStmt {{.*}} <col:10, col:28>
// CHECK-NEXT:        `-IfStmt {{.*}} <col:17, col:28>
// CHECK-NEXT:          |-CXXBoolLiteralExpr {{.*}} <col:21> 'bool' true
// CHECK-NEXT:          `-CompoundStmt {{.*}} <col:27, col:28>
