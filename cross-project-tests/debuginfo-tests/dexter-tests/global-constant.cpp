// REQUIRES: system-windows
//
// RUN: %clang_cl /Z7 /Zi %s -o %t
// RUN: %dexter --fail-lt 1.0 -w --binary %t --debugger 'dbgeng' -- %s

// Check that global constants have debug info.

const float TestPi = 3.14;
struct S {
  static const char TestCharA = 'a';
};
enum TestEnum : int {
  ENUM_POS = 2147000000,
  ENUM_NEG = -2147000000,
};
void useConst(int) {}
int main() {
  useConst(TestPi);
  useConst(S::TestCharA);
  useConst(ENUM_NEG); // DexLabel('stop')
  return 0;
}

// DexExpectWatchValue('TestPi', 3.140000104904175, on_line=ref('stop'))
// DexExpectWatchValue('S::TestCharA', 97, on_line=ref('stop'))
// DexExpectWatchValue('ENUM_NEG', -2147000000, on_line=ref('stop'))
/* DexExpectProgramState({'frames': [{
               'location': {'lineno' : ref('stop')},
               'watches': {'ENUM_POS' : {'is_irretrievable': True}}
}]}) */
