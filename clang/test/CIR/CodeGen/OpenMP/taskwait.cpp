// TODO: fix crash in emitTaskWaitCall
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fopenmp-enable-irbuilder -fopenmp -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// CHECK: cir.func
void omp_taskwait_1(){
// CHECK-DISABLE: omp.taskwait
//  #pragma omp taskwait
}
