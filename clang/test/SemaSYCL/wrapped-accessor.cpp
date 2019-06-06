// RUN: %clang_cc1 -I %S/Inputs -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct kernel wrapper in case when
// accessor is wrapped.

#include <sycl.hpp>

template <typename Acc>
struct AccWrapper { Acc accessor; };

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> acc;
  auto acc_wrapped = AccWrapper<decltype(acc)>{acc};
  kernel<class wrapped_access>(
      [=]() {
        acc_wrapped.accessor.use();
      });
}

// Check declaration of the kernel
// CHECK: wrapped_access 'void (AccWrapper<cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> >, __global int *, range<1>, range<1>, id<1>)'

// Check parameters of the kernel
// CHECK: ParmVarDecl {{.*}} used _arg_ 'AccWrapper<cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> >':'AccWrapper<cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> >'
// CHECK: ParmVarDecl {{.*}} used _arg_accessor '__global int *'
// CHECK: ParmVarDecl {{.*}} used [[_arg_AccessRange:[0-9a-zA-Z_]+]] 'range<1>':'cl::sycl::range<1>'
// CHECK: ParmVarDecl {{.*}} used [[_arg_MemRange:[0-9a-zA-Z_]+]] 'range<1>':'cl::sycl::range<1>'
// CHECK: ParmVarDecl {{.*}} used [[_arg_Offset:[0-9a-zA-Z_]+]] 'id<1>':'cl::sycl::id<1>'

// Check that wrapper object itself is initialized with corresponding kernel argument using operator=
// CHECK: BinaryOperator {{.*}} 'AccWrapper<cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> >':'AccWrapper<cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> >' lvalue '='

// Left operand is the field of the kernel object
// CHECK-NEXT: MemberExpr {{.*}} 'AccWrapper<cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> >':'AccWrapper<cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> >' lvalue . {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}wrapped-accessor.cpp{{.*}})' lvalue Var {{.*}} '(lambda at {{.*}}wrapped-accessor.cpp{{.*}})'

// Right operand is the kernel argument
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'AccWrapper<cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> >':'AccWrapper<cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> >' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'AccWrapper<cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> >':'AccWrapper<cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> >' lvalue ParmVar {{.*}} '_arg_' 'AccWrapper<cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> >':'AccWrapper<cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> >'

// Check that accessor field of the wrapper object is initialized using __init method
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global int *, range<1>, range<1>, id<1>)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t>':'cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t>' lvalue .accessor {{.*}}
// CHECK-NEXT: MemberExpr {{.*}} 'AccWrapper<cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> >':'AccWrapper<cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> >' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}wrapped-accessor.cpp{{.*}})' lvalue Var {{.*}} '(lambda at {{.*}}wrapped-accessor.cpp{{.*}})'

// Parameters of the _init method
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '_arg_accessor' '__global int *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'range<1>':'cl::sycl::range<1>' <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'range<1>':'cl::sycl::range<1>' lvalue ParmVar {{.*}} '[[_arg_AccessRange]]' 'range<1>':'cl::sycl::range<1>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'range<1>':'cl::sycl::range<1>' <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'range<1>':'cl::sycl::range<1>' lvalue ParmVar {{.*}} '[[_arg_MemRange]]' 'range<1>':'cl::sycl::range<1>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'id<1>':'cl::sycl::id<1>' <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'id<1>':'cl::sycl::id<1>' lvalue ParmVar {{.*}} '[[_arg_Offset]]' 'id<1>':'cl::sycl::id<1>'
