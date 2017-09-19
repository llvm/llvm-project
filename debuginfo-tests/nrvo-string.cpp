// This ensures that DW_OP_deref is inserted when necessary, such as when NRVO
// of a string object occurs in C++.
//
// RUN: %clangxx -O0 -fno-exceptions %target_itanium_abi_host_triple %s -o %t.out -g
// RUN: %test_debuginfo %s %t.out
//
// PR34513

struct string {
  string() {}
  string(int i) : i(i) {}
  ~string() {}
  int i = 0;
};
string get_string() {
  string unused;
  string result = 3;
// DEBUGGER: break 21
  return result;
}
int main() { get_string(); }

// DEBUGGER: r
// DEBUGGER: print result.i
// CHECK:  = 3
