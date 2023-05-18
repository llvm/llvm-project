// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -emit-llvm -o %t %s
// RUN: grep 'float 0x7FF8000000000000, float 0x7FF8000000000000, float 0x7FF8000020000000, float 0x7FF8000000000000, float 0x7FF80001E0000000, float 0x7FF8001E00000000, float 0x7FF801E000000000, float 0x7FF81E0000000000, float 0x7FF9E00000000000, float 0x7FFFFFFFE0000000' %t

float n[] = {
  __builtin_nanf("0"),
  __builtin_nanf(""),
  __builtin_nanf("1"),
  __builtin_nanf("0x7fc00000"),
  __builtin_nanf("0x7fc0000f"),
  __builtin_nanf("0x7fc000f0"),
  __builtin_nanf("0x7fc00f00"),
  __builtin_nanf("0x7fc0f000"),
  __builtin_nanf("0x7fcf0000"),
  __builtin_nanf("0xffffffff"),
};
