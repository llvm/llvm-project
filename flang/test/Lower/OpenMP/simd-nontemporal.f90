! Test nontemporal clause
! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
! RUN: bbc -emit-fir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s


! CHECK-LABEL: func @_QPsimd_with_nontemporal_clause
subroutine simd_with_nontemporal_clause(n)
  ! CHECK: %[[A_DECL:.*]] = fir.declare %{{.*}} {uniq_name = "_QFsimd_with_nontemporal_clauseEa"} : (!fir.ref<i32>) -> !fir.ref<i32>
  ! CHECK: %[[C_DECL:.*]] = fir.declare %{{.*}} {uniq_name = "_QFsimd_with_nontemporal_clauseEc"} : (!fir.ref<i32>) -> !fir.ref<i32>
  integer :: i, n
  integer :: A, B, C
  ! CHECK: omp.simd nontemporal(%[[A_DECL]], %[[C_DECL]] : !fir.ref<i32>, !fir.ref<i32>) private(@_QFsimd_with_nontemporal_clauseEi_private_i32 %8 -> %arg1 : !fir.ref<i32>) {
  ! CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
  !$OMP SIMD NONTEMPORAL(A, C)
  do i = 1, n
    ! CHECK:  %[[LOAD:.*]] = fir.load %[[A_DECL]] {nontemporal} : !fir.ref<i32>
    C = A + B
    ! CHECK: %[[ADD_VAL:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
    ! CHECK: fir.store %[[ADD_VAL]] to %[[C_DECL]] {nontemporal} : !fir.ref<i32>
  end do
  !$OMP END SIMD
end subroutine

! CHECK-LABEL:  func.func @_QPsimd_nontemporal_allocatable
subroutine simd_nontemporal_allocatable(x, y)
  integer, allocatable :: x(:)
  integer :: y
  allocate(x(100))
  ! CHECK:  %[[X_DECL:.*]] = fir.declare %{{.*}} dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, 
  ! CHECK-SAME: uniq_name = "_QFsimd_nontemporal_allocatableEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK:  omp.simd nontemporal(%[[X_DECL]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) private(@_QFsimd_nontemporal_allocatableEi_private_i32 %2 -> %arg2 : !fir.ref<i32>) {
  ! CHECK:   omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
  !$omp simd nontemporal(x)
  do i=1,100
    ! CHECK:  %[[VAL1:.*]] = fir.load %[[X_DECL]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    ! CHECK:  %[[BOX_ADDR:.*]] = fir.box_addr %[[VAL1]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    ! CHECK:  %[[ARR_COOR:.*]] = fir.array_coor %[[BOX_ADDR]](%{{.*}}) %{{.*}} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>, i64) -> !fir.ref<i32>
    ! CHECK:  %[[VAL2:.*]] = fir.load %[[ARR_COOR]] {nontemporal} : !fir.ref<i32>
  x(i) = x(i) + y
    ! CHECK:  fir.store %{{.*}} to %{{.*}} {nontemporal} : !fir.ref<i32>
  end do
  !$omp end simd
end subroutine

! CHECK-LABEL:  func.func @_QPsimd_nontemporal_pointers
subroutine simd_nontemporal_pointers(a, c)
   integer :: b, i
   integer :: n
   integer, pointer, intent(in):: a(:)
   integer, pointer, intent(out) :: c(:)
   ! CHECK:  %[[A_DECL:.*]] = fir.declare  %{{.*}} dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<intent_in, pointer>, 
   ! CHECK-SAME: uniq_name = "_QFsimd_nontemporal_pointersEa"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
   ! CHECK:  %[[C_DECL:.*]] = fir.declare %{{.*}} dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<intent_out, pointer>, 
   ! CHECK-SAME: uniq_name = "_QFsimd_nontemporal_pointersEc"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
   !$OMP SIMD NONTEMPORAL(a,c)
   do i = 1, n
      ! CHECK: %[[VAL1:.*]] = fir.load %[[A_DECL]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
      ! CHECK: %[[VAL2:.*]] = fir.array_coor %[[VAL1]](%{{.*}}) %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>, i64) -> !fir.ref<i32>
      ! CHECK: %[[VAL3:.*]] = fir.load %[[VAL2]] {nontemporal} : !fir.ref<i32>
      c(i) = a(i) + b
      ! CHECK: %[[VAL4:.*]] = fir.load %[[C_DECL]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
      ! CHECK: %[[VAL5:.*]] = fir.array_coor %[[VAL4]](%{{.*}}) %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>, i64) -> !fir.ref<i32>
      ! CHECK: fir.store %{{.*}} to %[[VAL5]] {nontemporal} : !fir.ref<i32>
   end do
   !$OMP END SIMD
end subroutine

