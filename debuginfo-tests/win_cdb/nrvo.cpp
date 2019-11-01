// This ensures that DW_OP_deref is inserted when necessary, such as when NRVO
// of a string object occurs in C++.
//
// RUN: %clang_cl %s -o %t.exe -fuse-ld=lld -Z7
// RUN: grep DE[B]UGGER: %s | sed -e 's/.*DE[B]UGGER: //' > %t.script
// RUN: %cdb -cf %t.script %t.exe | FileCheck %s --check-prefixes=DEBUGGER,CHECK
//

struct string {
  string() {}
  string(int i) : i(i) {}
  ~string() {}
  int i = 0;
};
string get_string() {
  string unused;
  string result = 3;
  __debugbreak();
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
  __debugbreak();
  return result;
}
int main() {
  get_string();
  get_string2();
}

// DEBUGGER: g
// DEBUGGER: ?? result
// CHECK: struct string *
// CHECK:    +0x000 i : 0n3
// DEBUGGER: g
// DEBUGGER: ?? result
// CHECK: struct string2 *
// CHECK:    +0x000 i : 0n5
// DEBUGGER: q
