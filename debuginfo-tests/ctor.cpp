// RUN: %clangxx %target_itanium_abi_host_triple -O0 -g %s -c -o %t.o
// RUN: %clangxx %target_itanium_abi_host_triple %t.o -o %t.out
// RUN: %test_debuginfo %s %t.out 


// DEBUGGER: break 14
// DEBUGGER: r
// DEBUGGER: p *this
// CHECK-NEXT-NOT: Cannot access memory at address 

class A {
public:
	A() : zero(0), data(42)
	{
	}
private:
	int zero;
	int data;
};

int main() {
	A a;
	return 0;
}

