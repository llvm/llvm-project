// RUN: %clang -g -std=c++2a -target x86_64-windows-msvc -Wno-gnu-string-literal-operator-template %s -S -emit-llvm -o - | FileCheck %s

template <typename T, T... cs> struct check;
template <typename T, T... str> int operator""_x() {
  return 1;
}

int b = u8"\"тест_𐀀"_x;
// CHECK: _x<char8_t,34,209,130,208,181,209,129,209,130,95,240,144,128,128>
