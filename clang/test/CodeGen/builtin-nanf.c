// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -emit-llvm -o %t %s
// RUN: grep 'float +qnan, float +qnan, float +nan(0x1), float +qnan, float +nan(0xF), float +nan(0xF0), float +nan(0xF00), float +nan(0xF000), float +nan(0xF0000), float +nan(0x3FFFFF)' %t

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
