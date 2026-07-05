// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

// CHECK: @tentative_attr_first ={{.*}} global i32 undef
int tentative_attr_first __attribute__((loader_uninitialized));
int tentative_attr_first;

// CHECK: @tentative_attr_second ={{.*}} global i32 undef
int tentative_attr_second;
int tentative_attr_second __attribute__((loader_uninitialized));

// CHECK: @array ={{.*}} global [16 x float] undef
float array[16] __attribute__((loader_uninitialized));

typedef struct
{
  int x;
  float y;
} s;

// CHECK: @i ={{.*}} global %struct.s undef
// CHECK: @j1 ={{.*}}addrspace(1) global %struct.s undef
// CHECK: @j2 ={{.*}}addrspace(2) global %struct.s undef
// CHECK: @j3 ={{.*}}addrspace(3) global %struct.s undef
// CHECK: @j4 ={{.*}}addrspace(4) global %struct.s undef
// CHECK: @j5 ={{.*}}addrspace(5) global %struct.s undef
// CHECK: @j99 ={{.*}}addrspace(99) global %struct.s undef
s i __attribute__((loader_uninitialized));
s j1 __attribute__((loader_uninitialized, address_space(1)));
s j2 __attribute__((loader_uninitialized, address_space(2)));
s j3 __attribute__((loader_uninitialized, address_space(3)));
s j4 __attribute__((loader_uninitialized, address_space(4)));
s j5 __attribute__((loader_uninitialized, address_space(5)));
s j99 __attribute__((loader_uninitialized, address_space(99)));

// CHECK: @private_extern_ok = hidden global i32 undef
__private_extern__ int private_extern_ok __attribute__((loader_uninitialized));
