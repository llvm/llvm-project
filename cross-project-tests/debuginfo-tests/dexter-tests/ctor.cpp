// REQUIRES: lldb
// UNSUPPORTED: system-windows
//
// RUN: %clang++ -std=gnu++11 -O0 -glldb %s -o %t
// RUN: %dexter -w --use-script \
// RUN:     --binary %t %dexter_lldb_args -- %s | FileCheck %s

class A {
public:
	A() : zero(0), data(42) { // !dex_label ctor_start
	}
private:
	int zero;
	int data;
};

int main() {
	A a;
	return 0;
}

// CHECK-DAG: total_watched_steps: 1
// CHECK-DAG: irretrievable_steps: 0

/*
---
!where {lines: !label ctor_start}:
  !value this:
    "*": "{}"
...
*/
