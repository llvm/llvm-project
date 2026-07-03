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

// We should step on ctor_start 1 (or more) times, and `this` should not be
// irretrievable when we do so.
// CHECK-DAG: total_watched_steps: {{[1-9]}}
// CHECK-DAG: irretrievable_steps: 0

/*
---
!where {lines: !label ctor_start}:
  !value this:
    "*": "{}"
...
*/
