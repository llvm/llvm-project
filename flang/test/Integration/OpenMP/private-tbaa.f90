! RUN: %flang_fc1 -O1 -fopenmp -emit-llvm %s -o - | FileCheck %s
! RUN: %flang_fc1 -O1 -fopenmp -mmlir --enable-delayed-privatization=false -emit-llvm %s -o - | FileCheck %s

! Both privatization paths must retain the allocation tag for the private
! automatic array through OpenMP outlining and LLVM IR optimization.
subroutine private_array(n)
  integer :: n
  real(8) :: tmp(5,n)
  !$omp parallel private(tmp)
    tmp(1,1) = 1.0_8
    call consume(tmp, n)
  !$omp end parallel
end subroutine

! CHECK: define internal void @{{.*}}omp_par
! CHECK: store double 1.000000e+00, ptr {{.*}}, align {{[0-9]+}}, !tbaa ![[TAG:[0-9]+]]
! CHECK-DAG: ![[TAG]] = !{![[TYPE:[0-9]+]], ![[TYPE]], i64 0}
! CHECK-DAG: ![[TYPE]] = !{!"allocated data/_QFprivate_arrayEtmp", ![[ALLOC:[0-9]+]], i64 0}
! CHECK-DAG: ![[ALLOC]] = !{!"allocated data", {{.*}}}
