// RUN: %clang_cc1 -triple avr -emit-llvm < %s | FileCheck %s

// Test that function declarations in nonzero address spaces without prototype
// are called correctly.

// CHECK: define void @bar() addrspace(1)
// CHECK: call addrspace(1) void bitcast (void (...) addrspace(1)* @foo to void (i16) addrspace(1)*)(i16 3)
// CHECK: declare void @foo(...) addrspace(1)
void foo();
void bar(void) {
	foo(3);
}
