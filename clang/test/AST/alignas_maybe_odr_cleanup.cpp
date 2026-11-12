// Test without serialization:
// RUN: %clang_cc1 %s -ast-dump | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s

struct FOO {
  static const int vec_align_bytes = 16;
  void foo() {
    double a alignas(vec_align_bytes);
    ;
  }
};

// CHECK:      |   `-AlignedAttr {{.*}} <col:14> alignas
// CHECK-NEXT:      |     `-ConstantExpr {{.*}} <col:22> 'int'
// CHECK-NEXT:      |       |-value: Int 16
// CHECK-NEXT:      |       `-ImplicitCastExpr {{.*}} <col:22> 'int' <LValueToRValue>
// CHECK-NEXT:      |         `-DeclRefExpr {{.*}} <col:22> 'const int' lvalue Var {{.*}} 'vec_align_bytes' 'const int' non_odr_use_constant
// CHECK-NEXT:      `-NullStmt {{.*}} <line:14:5>
