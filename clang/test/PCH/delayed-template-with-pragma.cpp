// RUN: %clang_cc1 -fopenmp -emit-pch -o %t.pch %s
// RUN: %clang_cc1 -fopenmp -fdelayed-template-parsing -emit-pch -o %t.delayed.pch %s
// RUN: %clang_cc1 -DMAIN_FILE -fopenmp -include-pch %t.pch \
// RUN:   -emit-llvm -o - %s -fopenmp | FileCheck %s
// RUN: %clang_cc1 -DMAIN_FILE -fopenmp -fdelayed-template-parsing -verify \
// RUN:   -Wunused-variable -include-pch %t.delayed.pch \
// RUN:   -emit-llvm -o - %s | FileCheck %s

#ifndef MAIN_FILE
template <typename T>
void a(T t) {
  #pragma clang loop unroll_count(4)
  for(int i=0;i<8;++i) {}
  #pragma omp simd
  for(int i=0;i<8;++i) {}
  {
    int x, y, z, zz;
    #pragma unused(x)
    #pragma unused(y, z)
  }
}
#else
// CHECK: !llvm.loop !3
// CHECK: !llvm.loop !7
// CHECK: !3 = distinct !{!3, !4, !5}
// CHECK: !4 = !{!"llvm.loop.mustprogress"}
// CHECK: !5 = !{!"llvm.loop.unroll.count", i32 4}
// CHECK: !7 = distinct !{!7, !8, !9}
// CHECK: !8 = !{!"llvm.loop.parallel_accesses", !6}
// CHECK: !9 = !{!"llvm.loop.vectorize.enable", i1 true}
// expected-warning@17 {{unused variable 'zz'}}
void foo()
{
  a(1);
}
#endif
