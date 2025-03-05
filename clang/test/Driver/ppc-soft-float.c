// RUN: %clang -target powerpc64-unknown-unknown -mcpu=pwr10 -msoft-float -S -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECKSOFT
// RUN: %clang -target powerpc64-unknown-unknown -mcpu=pwr10 -S -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECKNOSOFT

int main () {
  return 0;
}

// CHECKSOFT-DAG: -hard-float
// CHECKSOFT-DAG: -vsx
// CHECKSOFT-DAG: -altivec
// CHECKSOFT-DAG: -direct-move
// CHECKSOFT-DAG: -float128
// CHECKSOFT-DAG: -mma
// CHECKSOFT-DAG: -paired-vector-memops
// CHECKSOFT-DAG: -power10-vector
// CHECKSOFT-DAG: -power9-vector
// CHECKSOFT-DAG: -power8-vector
// CHECKSOFT-DAG: -crypto

// CHECKNOSOFT-DAG: +vsx
// CHECKNOSOFT-DAG: +altivec
// CHECKNOSOFT-DAG: +direct-move
// CHECKNOSOFT-DAG: +mma
// CHECKNOSOFT-DAG: +paired-vector-memops
// CHECKNOSOFT-DAG: +power10-vector
// CHECKNOSOFT-DAG: +power9-vector
// CHECKNOSOFT-DAG: +power8-vector
// CHECKNOSOFT-DAG: +crypto
