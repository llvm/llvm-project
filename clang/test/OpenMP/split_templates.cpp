// Dependent template defers transformation; explicit instantiation emits IR.
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -x c++ -std=c++17 -fopenmp -fopenmp-version=60 -ast-dump -DTEST_DEP %s | FileCheck %s --check-prefix=DEP
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -x c++ -std=c++17 -fopenmp -fopenmp-version=60 -O0 -emit-llvm -DTEST_INST %s -o - | FileCheck %s --check-prefix=LLVM

extern "C" void body(int);

#if defined(TEST_DEP)
template <typename T>
void dep_split(T n) {
#pragma omp split counts(2, omp_fill)
  for (T i = 0; i < n; ++i)
    body((int)i);
}
// DEP-LABEL: dep_split
// DEP: OMPSplitDirective
// DEP: ForStmt
#endif

#if defined(TEST_INST)
template <typename T>
void dep_split(T n) {
#pragma omp split counts(2, omp_fill)
  for (T i = 0; i < n; ++i)
    body((int)i);
}
template void dep_split<int>(int);
// LLVM: .split.iv
// LLVM: call void @body
#endif
