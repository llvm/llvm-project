// This ensures that DW_OP_deref is inserted when necessary, such as when NRVO
// of a string object occurs in C++.
//
// RUN: %clangxx -O0 -fno-exceptions %target_itanium_abi_host_triple %s -o %t.out -g
// RUN: %test_debuginfo %s %t.out
// RUN: %clangxx -O1 -fno-exceptions %target_itanium_abi_host_triple %s -o %t.out -g
// RUN: %test_debuginfo %s %t.out
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
string get_string() {
  string unused;
  string result = 3;
  // DEBUGGER: break 23
  stop();
  return result;
}
void some_function(int) {}
struct string2 {
  string2() = default;
  string2(string2 &&other) { i = other.i; }
  int i;
};
string2 get_string2() {
  string2 result;
  result.i = 5;
  some_function(result.i);
  // Test that the debugger can get the value of result after another
  // function is called.
  // DEBUGGER: break 39
  stop();
  return result;
}
int main() {
  get_string();
  get_string2();
}

// DEBUGGER: r
// DEBUGGER: print result.i
// CHECK:  = 3
// DEBUGGER: c
// DEBUGGER: print result.i
// CHECK:  = 5
