// RUN: %clang_cc1 -I %S/Inputs -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct initialization for arguments
// that have struct or built-in type inside the kernel wrapper

#include <sycl.hpp>

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

struct test_struct {
  int data;
};

int main() {
  int data = 5;
  test_struct s;
  s.data = data;
  kernel<class kernel_int>(
      [=]() {
        int kernel_data = data;
      });
  kernel<class kernel_struct>(
      [=]() {
        test_struct k_s;
        k_s = s;
      });
  return 0;
}
// Check kernel wrapper parameters
// CHECK: {{.*}}kernel_int 'void (int)'
// CHECK: ParmVarDecl {{.*}} used _arg_ 'int'

// Check that lambda field of built-in type is initialized with binary '='
// operator i.e lambda.field = _arg_
// CHECK: BinaryOperator {{.*}} 'int' lvalue '='
// CHECK-NEXT: MemberExpr {{.*}} 'int' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})' lvalue Var
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_' 'int'

// Check kernel wrapper parameters
// CHECK: {{.*}}kernel_struct 'void (test_struct)'
// CHECK: ParmVarDecl {{.*}} used _arg_ 'test_struct'

// Check that lambda field of struct type is initialized with binary '='
// operator i.e lambda.field = _arg_
// CHECK: BinaryOperator {{.*}} 'test_struct' lvalue '='
// CHECK-NEXT: MemberExpr {{.*}} 'test_struct' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})' lvalue Var
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'test_struct' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'test_struct' lvalue ParmVar {{.*}} '_arg_' 'test_struct'
