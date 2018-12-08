// We do not expect any error when one part of the triple is unknown, but other
// known parts are equal.

// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/ctudir/ctu-other.cpp.ast %S/Inputs/ctu-other.cpp
// RUN: cp %S/Inputs/ctu-other.cpp.externalFnMap.txt %t/ctudir/externalFnMap.txt
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -Werror=ctu \
// RUN:   -verify %s

// expected-no-diagnostics

int f(int);

int main() {
  return f(5);
}
