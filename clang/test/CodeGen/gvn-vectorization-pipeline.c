// RUN: %clang -O2 -S -emit-llvm -o /dev/null -mllvm -print-pipeline-passes %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -O3 -S -emit-llvm -o /dev/null -mllvm -print-pipeline-passes %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -O2 -fno-vectorize -S -emit-llvm -o /dev/null -mllvm -print-pipeline-passes %s 2>&1 | FileCheck %s --check-prefix=DISABLED --implicit-check-not=preserve-vectorization
// RUN: %clang -O3 -fno-vectorize -S -emit-llvm -o /dev/null -mllvm -print-pipeline-passes %s 2>&1 | FileCheck %s --check-prefix=DISABLED --implicit-check-not=preserve-vectorization
// RUN: %clang -O2 -flto=thin -S -emit-llvm -o /dev/null -mllvm -print-pipeline-passes %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -O2 -flto=full -S -emit-llvm -o /dev/null -mllvm -print-pipeline-passes %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -O2 -fno-vectorize -flto=thin -S -emit-llvm -o /dev/null -mllvm -print-pipeline-passes %s 2>&1 | FileCheck %s --check-prefix=DISABLED --implicit-check-not=preserve-vectorization
// RUN: %clang -O2 -fno-vectorize -flto=full -S -emit-llvm -o /dev/null -mllvm -print-pipeline-passes %s 2>&1 | FileCheck %s --check-prefix=DISABLED --implicit-check-not=preserve-vectorization

// Preserve vectorization opportunities only when loop vectorization is enabled
// later in the compilation, including vectorization deferred until ThinLTO.
// ENABLED: gvn<preserve-vectorization;>
// DISABLED: gvn<>
void f(void) {}
