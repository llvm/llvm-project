// RUN: %clang_cl %s -o %t.exe -fuse-ld=lld -Z7
// RUN: grep DE[B]UGGER: %s | sed -e 's/.*DE[B]UGGER: //' > %t.script
// RUN: %cdb -cf %t.script %t.exe | FileCheck %s --check-prefixes=DEBUGGER,CHECK

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
  useConst(ENUM_NEG);
  // DEBUGGER: g
  // DEBUGGER: ?? TestPi
  // CHECK: float 3.140000105
  // DEBUGGER: ?? S::TestCharA
  // CHECK: char 0n97 'a'
  // DEBUGGER: ?? ENUM_NEG
  // CHECK: TestEnum ENUM_NEG (0n-2147000000)
  // Unused constants shouldn't show up in the globals stream.
  // DEBUGGER: ?? ENUM_POS
  // CHECK: Couldn't resolve error at 'ENUM_POS'
  // DEBUGGER: q
  __debugbreak();
  return 0;
}
