// Purpose:
//     This ensures that DW_OP_deref is inserted when necessary, such as when
//     NRVO of a string object occurs in C++.
//
// REQUIRES: !asan, compiler-rt, lldb
// UNSUPPORTED: system-windows
//           Zorg configures the ASAN stage2 bots to not build the asan
//           compiler-rt. Only run this test on non-asanified configurations.
//
// RUN: %clang++ -std=gnu++11 -O0 -glldb -fno-exceptions %s -o %t
// RUN: %dexter -w \
// RUN:     --binary %t %dexter_lldb_args -- %s | FileCheck %s
//
// RUN: %clang++ -std=gnu++11 -O1 -glldb -fno-exceptions %s -o %t
// RUN: %dexter -w \
// RUN:     --binary %t %dexter_lldb_args -- %s | FileCheck %s
//
// PR34513
volatile int sideeffect = 0;
void __attribute__((noinline)) stop() { sideeffect++; }

struct string {
  string() {}
  string(int i) : i(i) {}
  ~string() {}
  int i = 0;
};
string __attribute__((noinline)) get_string() {
  string unused;
  string output = 3;
  stop(); // !dex_label string-nrvo
  return output;
}
void some_function(int) {}
struct string2 {
  string2() = default;
  string2(string2 &&other) { i = other.i; }
  int i;
};
string2 __attribute__((noinline)) get_string2() {
  string2 output;
  output.i = 5;
  some_function(output.i);
  // Test that the debugger can get the value of output after another
  // function is called.
  stop(); // !dex_label string2-nrvo
  return output;
}
int main() {
  get_string();
  get_string2();
}

// CHECK-DAG: seen_values: 2
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {lines: !label string-nrvo}:
  !value output:
    i: 3
!where {lines: !label string2-nrvo}:
  !value output:
    i: 5
...
*/
