// RUN: %clang_cc1 %s -ffake-address-space-map -faddress-space-map-mangling=no -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s

void func(local int*);

__kernel void foo(void) {
  // CHECK: @foo.i = internal addrspace(3) global i32 undef
  __local int i;
  func(&i);
}

// CHECK-LABEL: define {{.*}}void @_Z3barPU7CLlocali
__kernel void __attribute__((__overloadable__)) bar(local int *x) {
  *x = 5;
}
