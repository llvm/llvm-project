! This test checks lowering of OpenMP parallel DO, with the loop bound being
! a firstprivate variable

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s

! CHECK: func @_QPomp_do_firstprivate(%[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}) 
subroutine omp_do_firstprivate(a)
  integer::a
  integer::n
  n = a+1
  !$omp parallel do firstprivate(a)
  ! CHECK:  omp.parallel {
  ! CHECK-NEXT: %[[REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
  ! CHECK-NEXT: %[[CLONE:.*]] = fir.alloca i32 {bindc_name = "a", pinned
  ! CHECK-NEXT: %[[LD:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
  ! CHECK-NEXT: fir.store %[[LD]] to %[[CLONE]] : !fir.ref<i32>
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: %[[UB:.*]] = fir.load %[[CLONE]] : !fir.ref<i32>
  ! CHECK-NEXT: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: omp.wsloop   for  (%[[ARG1:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]])
  ! CHECK-NEXT: fir.store %[[ARG1]] to %[[REF]] : !fir.ref<i32>
  ! CHECK-NEXT: fir.call @_QPfoo(%[[REF]], %[[CLONE]]) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>) -> ()
  ! CHECK-NEXT: omp.yield
    do i=1, a
      call foo(i, a)
    end do
  !$omp end parallel do
  !CHECK: fir.call @_QPbar(%[[ARG0]]) {{.*}}: (!fir.ref<i32>) -> ()
  call bar(a)
end subroutine omp_do_firstprivate

! CHECK: func @_QPomp_do_firstprivate2(%[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) 
subroutine omp_do_firstprivate2(a, n)
  integer::a
  integer::n
  n = a+1
  !$omp parallel do firstprivate(a, n)
  ! CHECK:  omp.parallel {
  ! CHECK-NEXT: %[[REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
  ! CHECK-NEXT: %[[CLONE:.*]] = fir.alloca i32 {bindc_name = "a", pinned
  ! CHECK-NEXT: %[[LD:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
  ! CHECK-NEXT: fir.store %[[LD]] to %[[CLONE]] : !fir.ref<i32>
  ! CHECK-NEXT: %[[CLONE1:.*]] = fir.alloca i32 {bindc_name = "n", pinned
  ! CHECK-NEXT: %[[LD1:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
  ! CHECK-NEXT: fir.store %[[LD1]] to %[[CLONE1]] : !fir.ref<i32>


  ! CHECK: %[[LB:.*]] = fir.load %[[CLONE]] : !fir.ref<i32>
  ! CHECK-NEXT: %[[UB:.*]] = fir.load %[[CLONE1]] : !fir.ref<i32>
  ! CHECK-NEXT: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: omp.wsloop   for  (%[[ARG2:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]])
  ! CHECK-NEXT: fir.store %[[ARG2]] to %[[REF]] : !fir.ref<i32>
  ! CHECK-NEXT: fir.call @_QPfoo(%[[REF]], %[[CLONE]]) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>) -> ()
  ! CHECK-NEXT: omp.yield
    do i= a, n
      call foo(i, a)
    end do
  !$omp end parallel do
  !CHECK: fir.call @_QPbar(%[[ARG1]]) {{.*}}: (!fir.ref<i32>) -> ()
  call bar(n)
end subroutine omp_do_firstprivate2
