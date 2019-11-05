// This ensures that DW_OP_deref is inserted when necessary, such as when NRVO
// of a string object occurs in C++.
//
// REQUIRES: system-windows
//
// RUN: %dexter --fail-lt 1.0 -w --builder 'clang-cl_vs2015' \
// RUN:      --debugger 'dbgeng' --cflags '/Z7 /Zi' --ldflags '/Z7 /Zi' -- %s

struct string {
  string() {}
  string(int i) : i(i) {}
  ~string() {}
  int i = 0;
};
string get_string() {
  string unused;
  string result = 3;
  return result; // DexLabel('readresult1')
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
  return result; // DexLabel('readresult2')
}
int main() {
  get_string();
  get_string2();
}

// DexExpectWatchValue('result.i', 3, on_line='readresult1')
// DexExpectWatchValue('result.i', 5, on_line='readresult2')
