// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-print-source-range-info %s 2>&1 | FileCheck %s

struct S {
  char a : 12 - 12;
};
// CHECK: misc-source-ranges.cpp:[[@LINE-2]]:8:{[[@LINE-2]]:12-[[@LINE-2]]:19}

