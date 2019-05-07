// RUN: %clang -S -I %S/Inputs --sycl -Xclang -ast-dump %s | FileCheck %s

#include <sycl.hpp>

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::sampler Sampler;
  kernel<class use_kernel_for_test>([=]() {
    Sampler.use();
  });
  return 0;
}

// Check declaration of the test kernel
// CHECK: FunctionDecl {{.*}}use_kernel_for_test 'void (__spirv::OpTypeSampler *)'
//
// Check parameters of the test kernel
// CHECK: ParmVarDecl {{.*}} used [[_arg_sampler:[0-9a-zA-Z_]+]] '__spirv::OpTypeSampler *'
//
// Check that sampler field of the test kernel object is initialized using __init method
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__spirv::OpTypeSampler *)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'cl::sycl::sampler':'cl::sycl::sampler' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}sampler.cpp{{.*}})' lvalue Var {{.*}} '(lambda at {{.*}}sampler.cpp{{.*}})'
//
// Check the parameters of __init method
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__spirv::OpTypeSampler *' <LValueToRValue>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__spirv::OpTypeSampler *' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} '__spirv::OpTypeSampler *' lvalue ParmVar {{.*}} '[[_arg_sampler]]' '__spirv::OpTypeSampler
