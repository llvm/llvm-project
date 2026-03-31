// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir
// RUN: %clang_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/ctudir/ctu-other.cpp.ast %S/Inputs/ctu-other.cpp
// RUN: cp %S/Inputs/ctu-other.cpp.externalDefMap.ast-dump.txt %t/ctudir/externalDefMap.txt
// RUN: %clang_analyze_cc1 -std=c++14 -triple powerpc64-montavista-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -Werror=ctu \
// RUN:   -verify %s

// We expect an error in this file, but without a location.
// expected-error-re@./ctu-different-triples.cpp:*{{imported AST from {{.*}} had been generated for a different target, current: powerpc64-montavista-linux-gnu, imported: x86_64-pc-linux-gnu}}

int f(int);

int main() {
  return f(5);
}
