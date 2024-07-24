// RUN: %clang -target powerpc64-unknown-unknown -mcpu=pwr10 -msoft-float -S -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECKSOFT
// RUN: %clang -target powerpc64-unknown-unknown -mcpu=pwr10 -S -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECKNOSOFT

int main () {
  return 0;
}

// CHECKSOFT-DAG: -hard-float
// CHECKSOFT-DAG: -vsx
// CHECKSOFT-DAG: -altivec

// CHECKNOSOFT-DAG: +vsx
// CHECKNOSOFT-DAG: +altivec
