// RUN: %clang_cc1 -triple avr -emit-llvm < %s | FileCheck %s

// CHECK: @var0 {{.*}} addrspace(1) constant [3 x i16]
// CHECK: @bar.var2 {{.*}} addrspace(1) constant [3 x i16]
// CHECK: @var1 {{.*}} addrspace(1) constant [3 x i16]

// CHECK: define{{.*}} void @bar() addrspace(1)
// CHECK: call addrspace(1) void bitcast (void (...) addrspace(1)* @foo to void (i16) addrspace(1)*)
// CHECK: declare void @foo(...) addrspace(1)

__flash const int var0[] = {999, 888, 777};
__flash static const int var1[] = {111, 222, 333};

int i;

void foo();

void bar() {
  static __flash const int var2[] = {555, 666, 777};
  foo(var1[i]);
}
