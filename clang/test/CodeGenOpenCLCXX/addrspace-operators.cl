//RUN: %clang_cc1 %s -triple spir -cl-std=c++ -emit-llvm -O0 -o - | FileCheck %s

enum E {
  a,
  b,
};

class C {
public:
  void Assign(E e) { me = e; }
  void OrAssign(E e) { mi |= e; }
  E me;
  int mi;
};

__global E globE;
//CHECK-LABEL: define spir_func void @_Z3barv()
void bar() {
  C c;
  //CHECK: addrspacecast %class.C* %c to %class.C addrspace(4)*
  //CHECK: call void @_ZNU3AS41C6AssignE1E(%class.C addrspace(4)* %{{[0-9]+}}, i32 0)
  c.Assign(a);
  //CHECK: addrspacecast %class.C* %c to %class.C addrspace(4)*
  //CHECK: call void @_ZNU3AS41C8OrAssignE1E(%class.C addrspace(4)* %{{[0-9]+}}, i32 0)
  c.OrAssign(a);

  E e;
  // CHECK: store i32 1, i32* %e
  e = b;
  // CHECK: store i32 0, i32 addrspace(1)* @globE
  globE = a;
  // FIXME: Sema fails here because it thinks the types are incompatible.
  //e = b;
  //globE = a;
}

//CHECK: define linkonce_odr void @_ZNU3AS41C6AssignE1E(%class.C addrspace(4)* %this, i32 %e)
//CHECK: [[E:%[0-9]+]] = load i32, i32* %e.addr
//CHECK: %me = getelementptr inbounds %class.C, %class.C addrspace(4)* %this1, i32 0, i32 0
//CHECK: store i32 [[E]], i32 addrspace(4)* %me

//CHECK define linkonce_odr void @_ZNU3AS41C8OrAssignE1E(%class.C addrspace(4)* %this, i32 %e)
//CHECK: [[E:%[0-9]+]] = load i32, i32* %e.addr
//CHECK: %mi = getelementptr inbounds %class.C, %class.C addrspace(4)* %this1, i32 0, i32 1
//CHECK: [[MI:%[0-9]+]] = load i32, i32 addrspace(4)* %mi
//CHECK: %or = or i32 [[MI]], [[E]]
