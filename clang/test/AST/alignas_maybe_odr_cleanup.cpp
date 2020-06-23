// Test without serialization:
// RUN: %clang_cc1 -fsyntax-only %s -ast-dump | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s

struct FOO {
  static const int vec_align_bytes = 32;
  void foo() {
    double a alignas(vec_align_bytes);
    ;
  }
};

// CHECK: AlignedAttr {{.*}} alignas
// CHECK: ConstantExpr {{.+}} 'int' Int: 32
// CHECK: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK: DeclRefExpr {{.*}} 'const int' lvalue Var {{.*}} 'vec_align_bytes' 'const int' non_odr_use_constant
// CHECK: NullStmt
