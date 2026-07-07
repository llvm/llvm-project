/* Non-int IV types with split. */
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=U32
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=I64

extern void body(unsigned int);
extern void body64(long);

// U32-LABEL: define {{.*}} @unsigned_iv
// U32: .split.iv
// U32-DAG: icmp ult i32
void unsigned_iv(void) {
#pragma omp split counts(2, omp_fill)
  for (unsigned i = 0; i < 10U; ++i)
    body(i);
}

// I64-LABEL: define {{.*}} @long_iv
// I64: .split.iv
// I64-DAG: icmp slt i64
void long_iv(void) {
#pragma omp split counts(2, omp_fill)
  for (long i = 0; i < 10L; ++i)
    body64(i);
}
