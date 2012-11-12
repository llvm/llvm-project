// RUN: %clang_cc1 %s -fms-extensions -triple i686-pc-win32 -emit-llvm -o - | FileCheck %s

// CHECK: @L__FUNCTION__._Z4funcv = private constant [5 x i16] [i16 102, i16 117, i16 110, i16 99, i16 0], align 2

void wprint(const wchar_t*);

#define __STR2WSTR(str) L##str
#define _STR2WSTR(str) __STR2WSTR(str)
#define STR2WSTR(str) _STR2WSTR(str)

void func() {
  wprint(STR2WSTR(__FUNCTION__));
}

int main() {
  func();

  return 0;
}

