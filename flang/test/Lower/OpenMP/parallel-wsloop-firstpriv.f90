! This test checks lowering of OpenMP parallel DO, with the loop bound being
! a firstprivate variable

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

! CHECK: func @_QPomp_do_firstprivate(%[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}) 
subroutine omp_do_firstprivate(a)
  ! CHECK: %[[ARG0_DECL:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFomp_do_firstprivateEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer::a
  integer::n
  n = a+1
  !$omp parallel do firstprivate(a)
  ! CHECK:  omp.parallel {
  ! CHECK: %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
  ! CHECK: %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFomp_do_firstprivateEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[A_PVT_REF:.*]] = fir.alloca i32 {bindc_name = "a", pinned, uniq_name = "_QFomp_do_firstprivateEa"}
  ! CHECK: %[[A_PVT_DECL:.*]]:2 = hlfir.declare %[[A_PVT_REF]] {uniq_name = "_QFomp_do_firstprivateEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK-NEXT: %[[LD:.*]] = fir.load %[[ARG0_DECL]]#1 : !fir.ref<i32>
  ! CHECK-NEXT: fir.store %[[LD]] to %[[A_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: %[[UB:.*]] = fir.load %[[A_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK-NEXT: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: omp.wsloop   for  (%[[ARG1:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]])
  ! CHECK-NEXT: fir.store %[[ARG1]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK-NEXT: fir.call @_QPfoo(%[[I_PVT_DECL]]#1, %[[A_PVT_DECL]]#1) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>) -> ()
  ! CHECK-NEXT: omp.yield
    do i=1, a
      call foo(i, a)
    end do
  !$omp end parallel do
  !CHECK: fir.call @_QPbar(%[[ARG0_DECL]]#1) {{.*}}: (!fir.ref<i32>) -> ()
  call bar(a)
end subroutine omp_do_firstprivate

! CHECK: func @_QPomp_do_firstprivate2(%[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) 
subroutine omp_do_firstprivate2(a, n)
  ! CHECK:  %[[ARG0_DECL:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFomp_do_firstprivate2Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK:  %[[ARG1_DECL:.*]]:2 = hlfir.declare %[[ARG1]] {uniq_name = "_QFomp_do_firstprivate2En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer::a
  integer::n
  n = a+1
  !$omp parallel do firstprivate(a, n)
  ! CHECK:  omp.parallel {
  ! CHECK: %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
  ! CHECK: %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFomp_do_firstprivate2Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[A_PVT_REF:.*]] = fir.alloca i32 {bindc_name = "a", pinned
  ! CHECK: %[[A_PVT_DECL:.*]]:2 = hlfir.declare %[[A_PVT_REF]] {uniq_name = "_QFomp_do_firstprivate2Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[LD:.*]] = fir.load %[[ARG0_DECL]]#1 : !fir.ref<i32>
  ! CHECK: fir.store %[[LD]] to %[[A_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK: %[[N_PVT_REF:.*]] = fir.alloca i32 {bindc_name = "n", pinned, uniq_name = "_QFomp_do_firstprivate2En"}
  ! CHECK: %[[N_PVT_DECL:.*]]:2 = hlfir.declare %[[N_PVT_REF]] {uniq_name = "_QFomp_do_firstprivate2En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[LD1:.*]] = fir.load %[[ARG1_DECL]]#1 : !fir.ref<i32>
  ! CHECK: fir.store %[[LD1]] to %[[N_PVT_DECL]]#1 : !fir.ref<i32>


  ! CHECK: %[[LB:.*]] = fir.load %[[A_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[UB:.*]] = fir.load %[[N_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: omp.wsloop   for  (%[[ARG2:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]])
  ! CHECK: fir.store %[[ARG2]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK: fir.call @_QPfoo(%[[I_PVT_DECL]]#1, %[[A_PVT_DECL]]#1) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>) -> ()
  ! CHECK: omp.yield
    do i= a, n
      call foo(i, a)
    end do
  !$omp end parallel do
  !CHECK: fir.call @_QPbar(%[[ARG1_DECL]]#1) {{.*}}: (!fir.ref<i32>) -> ()
  call bar(n)
end subroutine omp_do_firstprivate2
