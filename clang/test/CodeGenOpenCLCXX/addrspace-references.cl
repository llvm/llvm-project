//RUN: %clang_cc1 %s -cl-std=c++ -triple spir -emit-llvm -o - | FileCheck %s

int bar(const unsigned int &i);
// CHECK-LABEL: define spir_func void @_Z3foov() 
void foo() {
  // The generic addr space reference parameter object will be bound
  // to a temporary value allocated in private addr space. We need an
  // addrspacecast before passing the value to the function.
  // CHECK: [[REF:%.*]] = alloca i32
  // CHECK: store i32 1, i32* [[REF]]
  // CHECK: [[REG:%[0-9]+]] = addrspacecast i32* [[REF]] to i32 addrspace(4)*
  // CHECK: call spir_func i32 @_Z3barRU3AS4Kj(i32 addrspace(4)* dereferenceable(4) [[REG]])
  bar(1);
}
