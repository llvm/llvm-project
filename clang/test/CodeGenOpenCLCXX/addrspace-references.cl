//RUN: %clang_cc1 %s -cl-std=clc++ -triple spir -emit-llvm -o - -O0 | FileCheck %s

typedef short short2 __attribute__((ext_vector_type(2)));

int bar(const unsigned int &i);

class C {
public:
  void bar(const short2 &);
};

// CHECK-LABEL: define{{.*}} spir_func void @_Z6scalarv()
void scalar() {
  // The generic addr space reference parameter object will be bound
  // to a temporary value allocated in private addr space. We need an
  // addrspacecast before passing the value to the function.
  // CHECK: [[REF:%.*]] = alloca i32
  // CHECK: store i32 1, i32* [[REF]]
  // CHECK: [[REG:%[.a-z0-9]+]] ={{.*}} addrspacecast i32* [[REF]] to i32 addrspace(4)*
  // CHECK: call spir_func i32 @_Z3barRU3AS4Kj(i32 addrspace(4)* align 4 dereferenceable(4) [[REG]])
  bar(1);
}

// Test list initialization
// CHECK-LABEL: define{{.*}} spir_func void @_Z4listv()
void list() {
  C c1;
// CHECK: [[REF:%.*]] = alloca <2 x i16>
// CHECK: store <2 x i16> <i16 1, i16 2>, <2 x i16>* [[REF]]
// CHECK: [[REG:%[.a-z0-9]+]] = addrspacecast <2 x i16>* [[REF]] to <2 x i16> addrspace(4)*
// CHECK: call {{.*}}void @_ZNU3AS41C3barERU3AS4KDv2_s(%class.C addrspace(4)* {{.*}}, <2 x i16> addrspace(4)*{{.*}} [[REG]])
  c1.bar({1, 2});
}
