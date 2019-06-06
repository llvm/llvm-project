// RUN: %clang_cc1 -std=c++11 -I %S/Inputs -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct kernel wrapper for basic
// case.

#include <sycl.hpp>

template <typename Acc>
struct AccWrapper { Acc accessor; };

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> acc;
  kernel<class kernel_wrapper>(
      [=]() {
        acc.use();
      });
}

// Check declaration of the kernel

// CHECK: FunctionDecl {{.*}}kernel_wrapper 'void (__global int *, range<1>, range<1>, id<1>)'

// Check parameters of the kernel

// CHECK: ParmVarDecl {{.*}} used [[_arg_Mem:[0-9a-zA-Z_]+]] '__global int *'
// CHECK: ParmVarDecl {{.*}} used [[_arg_AccessRange:[0-9a-zA-Z_]+]] 'range<1>':'cl::sycl::range<1>'
// CHECK: ParmVarDecl {{.*}} used [[_arg_MemRange:[0-9a-zA-Z_]+]] 'range<1>':'cl::sycl::range<1>'
// CHECK: ParmVarDecl {{.*}} used [[_arg_Offset:[0-9a-zA-Z_]+]] 'id<1>':'cl::sycl::id<1>'

// Check body of the kernel

// Check lambda declaration inside the wrapper

// CHECK: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} used '(lambda at {{.*}}basic-kernel-wrapper.cpp{{.*}})'

// Check accessor initialization

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global int *, range<1>, range<1>, id<1>)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write>':'cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t>' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}basic-kernel-wrapper.cpp{{.*}})' lvalue Var
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '[[_arg_Mem]]' '__global int *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'range<1>':'cl::sycl::range<1>' <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'range<1>':'cl::sycl::range<1>' lvalue ParmVar {{.*}} '[[_arg_AccessRange]]' 'range<1>':'cl::sycl::range<1>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'range<1>':'cl::sycl::range<1>' <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'range<1>':'cl::sycl::range<1>' lvalue ParmVar {{.*}} '[[_arg_MemRange]]' 'range<1>':'cl::sycl::range<1>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'id<1>':'cl::sycl::id<1>' <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'id<1>':'cl::sycl::id<1>' lvalue ParmVar {{.*}} '[[_arg_Offset]]' 'id<1>':'cl::sycl::id<1>'

// Check that body of the kernel caller function is included into kernel

// CHECK: CompoundStmt {{.*}}
// CHECK-NEXT: CXXOperatorCallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const (lambda at {{.*}}basic-kernel-wrapper.cpp{{.*}})' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}basic-kernel-wrapper.cpp{{.*}})' lvalue Var

// Check kernel wrapper attributes

// CHECK: SYCLDeviceAttr {{.*}} Implicit
// CHECK: OpenCLKernelAttr {{.*}} Implicit
// CHECK: AsmLabelAttr {{.*}} Implicit "{{.*}}kernel_wrapper"
// CHECK: ArtificialAttr {{.*}} Implicit
