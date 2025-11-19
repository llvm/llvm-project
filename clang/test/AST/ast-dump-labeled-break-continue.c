// Test without serialization:
// RUN: %clang_cc1 -std=c2y -ast-dump %s \
// RUN: | FileCheck -strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -std=c2y -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -std=c2y -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck -strict-whitespace %s

void TestLabeledBreakContinue() {
  a: while (true) {
    break a;
    continue a;
    c: for (;;) {
      break a;
      continue a;
      break c;
    }
  }
}

// CHECK-LABEL: `-FunctionDecl {{.*}} TestLabeledBreakContinue
// CHECK-NEXT:     `-CompoundStmt {{.*}} <col:33, line:21:1>
// CHECK-NEXT:       `-LabelStmt {{.*}} <line:12:3, line:20:3> 'a'
// CHECK-NEXT:         `-WhileStmt {{.*}} <line:12:6, line:20:3>
// CHECK-NEXT:           |-CXXBoolLiteralExpr {{.*}} <line:12:13> 'bool' true
// CHECK-NEXT:           `-CompoundStmt {{.*}} <col:19, line:20:3>
// CHECK-NEXT:             |-BreakStmt {{.*}} <line:13:5, col:11> 'a' (WhileStmt {{.*}})
// CHECK-NEXT:             |-ContinueStmt {{.*}} <line:14:5, col:14> 'a' (WhileStmt {{.*}})
// CHECK-NEXT:             `-LabelStmt {{.*}} <line:15:5, line:19:5> 'c'
// CHECK-NEXT:               `-ForStmt {{.*}} <line:15:8, line:19:5>
// CHECK-NEXT:                 |-<<<NULL>>>
// CHECK-NEXT:                 |-<<<NULL>>>
// CHECK-NEXT:                 |-<<<NULL>>>
// CHECK-NEXT:                 |-<<<NULL>>>
// CHECK-NEXT:                 `-CompoundStmt {{.*}} <line:15:17, line:19:5>
// CHECK-NEXT:                   |-BreakStmt {{.*}} <line:16:7, col:13> 'a' (WhileStmt {{.*}})
// CHECK-NEXT:                   |-ContinueStmt {{.*}} <line:17:7, col:16> 'a' (WhileStmt {{.*}})
// CHECK-NEXT:                   `-BreakStmt {{.*}} <line:18:7, col:13> 'c' (ForStmt {{.*}})
