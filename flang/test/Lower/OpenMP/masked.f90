!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: func @_QPomp_masked
subroutine omp_masked(threadId)
integer :: threadId

!CHECK: omp.masked  {
!$omp masked

    !CHECK: fir.call @_QPmasked() {{.*}}: () -> ()
    call masked()

!CHECK: omp.terminator
!$omp end masked

!CHECK: omp.masked filter({{.*}})  {
!$omp masked filter(threadId)

    !CHECK: fir.call @_QPmaskedwithfilter() {{.*}}: () -> ()
    call maskedWithFilter()

!CHECK: omp.terminator
!$omp end masked
end subroutine omp_masked

