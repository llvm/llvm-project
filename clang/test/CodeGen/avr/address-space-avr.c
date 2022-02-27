// RUN: %clang_cc1 -triple avr -target-cpu atmega2560 -emit-llvm < %s | FileCheck %s

// CHECK: @var0 {{.*}} addrspace(1) constant [3 x i16]
// CHECK: @f3var0 {{.*}} addrspace(4) constant [3 x i16]
// CHECK: @bar.var2 {{.*}} addrspace(1) constant [3 x i16]
// CHECK: @bar.f3var2 {{.*}} addrspace(4) constant [3 x i16]
// CHECK: @var1 {{.*}} addrspace(1) constant [3 x i16]
// CHECK: @f3var1 {{.*}} addrspace(4) constant [3 x i16]

// CHECK: define{{.*}} void @bar() addrspace(1)
// CHECK: call addrspace(1) void bitcast (void (...) addrspace(1)* @foo to void (i16) addrspace(1)*)
// CHECK: declare void @foo(...) addrspace(1)

__flash const int var0[] = {999, 888, 777};
__flash static const int var1[] = {111, 222, 333};

__flash3 const int f3var0[] = {12, 34, 56};
__flash3 static const int f3var1[] = {52, 64, 96};

int i;

void foo();

void bar(void) {
  static __flash const int var2[] = {555, 666, 777};
  static __flash3 const int f3var2[] = {5555, 6666, 7787};
  foo(var1[i]);
  foo(f3var1[i]);
}
