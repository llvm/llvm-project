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
// CHECK: FunctionDecl {{.*}}use_kernel_for_test 'void (__ocl_sampler_t)'
//
// Check parameters of the test kernel
// CHECK: ParmVarDecl {{.*}} used [[_arg_sampler:[0-9a-zA-Z_]+]] '__ocl_sampler_t'
//
// Check that sampler field of the test kernel object is initialized using __init method
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__ocl_sampler_t)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'cl::sycl::sampler':'cl::sycl::sampler' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}sampler.cpp{{.*}})' lvalue Var {{.*}} '(lambda at {{.*}}sampler.cpp{{.*}})'
//
// Check the parameters of __init method
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__ocl_sampler_t':'sampler_t' <LValueToRValue>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__ocl_sampler_t':'sampler_t' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} '__ocl_sampler_t':'sampler_t' lvalue ParmVar {{.*}} '[[_arg_sampler]]' '__ocl_sampler_t':'sampler_t'
