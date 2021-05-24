// Purpose:
//     This ensures that DW_OP_deref is inserted when necessary, such as when
//     NRVO of a string object occurs in C++.
//
// REQUIRES: !asan, lldb
// UNSUPPORTED: system-windows
//           Zorg configures the ASAN stage2 bots to not build the asan
//           compiler-rt. Only run this test on non-asanified configurations.
//
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' \
// RUN:     --cflags "-O0 -glldb -fno-exceptions" -- %s
//
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' \
// RUN:     --cflags "-O1 -glldb -fno-exceptions" -- %s
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
  stop(); // DexLabel('string-nrvo')
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
  stop(); // DexLabel('string2-nrvo')
  return output;
}
int main() {
  get_string();
  get_string2();
}

// DexExpectWatchValue('output.i', 3, on_line=ref('string-nrvo'))
// DexExpectWatchValue('output.i', 5, on_line=ref('string2-nrvo'))

