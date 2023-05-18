// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-print-source-range-info -Wcast-function-type-strict %s 2>&1 | FileCheck %s

struct S {
  char a : 12 - 12;
};
// CHECK: misc-source-ranges.cpp:[[@LINE-2]]:8:{[[@LINE-2]]:12-[[@LINE-2]]:19}

using fun = long(*)(int &);
fun foo(){
  long (*f_ptr)(const int &);
  return fun(f_ptr);
}
// CHECK: misc-source-ranges.cpp:[[@LINE-2]]:10:{[[@LINE-2]]:10-[[@LINE-2]]:20}
