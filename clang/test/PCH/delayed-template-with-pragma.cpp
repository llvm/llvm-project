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
// CHECK: !llvm.loop [[LOOP1:!.*]]
// CHECK: !llvm.loop [[LOOP2:!.*]]
// CHECK: [[LOOP1]] = distinct !{[[LOOP1]], [[LOOP1A:!.*]], [[LOOP1B:!.*]]}
// CHECK: [[LOOP1A]] = !{!"llvm.loop.mustprogress"}
// CHECK: [[LOOP1B]] = !{!"llvm.loop.unroll.count", i32 4}
// CHECK: [[LOOP2]] = distinct !{[[LOOP2]], [[LOOP2A:!.*]], [[LOOP2B:!.*]]}
// CHECK: [[LOOP2A]] = !{!"llvm.loop.parallel_accesses", [[LOOP2C:!.*]]}
// CHECK: [[LOOP2B]] = !{!"llvm.loop.vectorize.enable", i1 true}
// expected-warning@17 {{unused variable 'zz'}}
void foo()
{
  a(1);
}
#endif
