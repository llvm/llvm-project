! REQUIRES: openmp_runtime
! RUN: bbc -emit-hlfir %openmp_flags -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir %openmp_flags -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL:  omp.private
! CHECK-SAME:       {type = private} @[[LAST_PRIVATE_I:.*]] : i32

! CHECK-LABEL:  omp.private
! CHECK-SAME:       {type = private} @[[LAST_PRIVATE_X:.*]] : i32

! CHECK-LABEL:  omp.private
! CHECK-SAME:       {type = private} @[[QFOMP_TASKLOOP_NOGROUPEI_PRIVATE_I32:.*]] : i32

! CHECK-LABEL:  omp.private
! CHECK-SAME:       {type = private} @[[OMP_TASKLOOP_UNTIEDEI_PRIVATE_I32:.*]] : i32

! CHECK-LABEL:  omp.private
! CHECK-SAME:       {type = private} @[[QFTEST_PRIORITYEI_PRIVATE_I32:.*]] : i32

! CHECK-LABEL:  omp.private
! CHECK-SAME:       {type = private} @[[QFTEST_MERGEABLEEI_PRIVATE_I32:.*]] : i32

! CHECK-LABEL:  omp.private
! CHECK-SAME:       {type = private} @[[I_PRIVATE_IF_TEST1:.*]] : i32

! CHECK-LABEL:  omp.private
! CHECK-SAME:       {type = private} @[[I_PRIVATE_FINAL:.*]] : i32

! CHECK-LABEL:  omp.private
! CHECK-SAME:       {type = private} @[[I_PRIVATE_TEST_ALLOCATE:.*]] : i32

! CHECK-LABEL:  omp.private
! CHECK-SAME:       {type = private} @[[X_PRIVATE_TEST_ALLOCATE:.*]] : i32

! CHECK-LABEL:  omp.private 
! CHECK-SAME:       {type = private} @[[I_PRIVATE_TEST2:.*]] : i32

! CHECK-LABEL:  omp.private 
! CHECK-SAME:       {type = private} @[[RES_PRIVATE_TEST2:.*]] : i32

! CHECK-LABEL:  omp.private 
! CHECK-SAME:       {type = private} @[[I_PRIVATE:.*]] : i32

! CHECK-LABEL:  omp.private 
! CHECK-SAME:        {type = firstprivate} @[[RES_FIRSTPRIVATE:.*]] : i32 
! CHECK-SAME:   copy {
! CHECK:         hlfir.assign 

! CHECK-LABEL:  func.func @_QPomp_taskloop
! CHECK:          %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_taskloopEi"}
! CHECK:          %[[I_VAL:.*]]:2 = hlfir.declare %[[ALLOCA_I]] {uniq_name = "_QFomp_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          %[[ALLOCA_RES:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFomp_taskloopEres"}
! CHECK:          %[[RES_VAL:.*]]:2 = hlfir.declare %[[ALLOCA_RES]] {uniq_name = "_QFomp_taskloopEres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:          %[[C10_I32:.*]] = arith.constant 10 : i32
! CHECK:          %[[C1_I32_0:.*]] = arith.constant 1 : i32
! CHECK:          omp.taskloop private(@[[RES_FIRSTPRIVATE]] %[[RES_VAL]]#0 -> %[[PRIV_RES:.*]], @[[I_PRIVATE]] %[[I_VAL]]#0 -> %[[PRIV_I:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:            omp.loop_nest (%[[ARG2:.*]]) : i32 = (%[[C1_I32]]) to (%[[C10_I32]]) inclusive step (%[[C1_I32_0]]) {
! CHECK:              %[[RES_DECL:.*]]:2 = hlfir.declare %[[PRIV_RES]] {uniq_name = "_QFomp_taskloopEres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:              %[[I_DECL:.*]]:2 = hlfir.declare %[[PRIV_I]] {uniq_name = "_QFomp_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:              hlfir.assign %[[ARG2]] to %[[I_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:              %[[LOAD_RES:.*]] = fir.load %[[RES_DECL]]#0 : !fir.ref<i32>
! CHECK:              %[[C1_I32_1:.*]] = arith.constant 1 : i32
! CHECK:              %[[OUT_VAL:.*]] = arith.addi %[[LOAD_RES]], %[[C1_I32_1]] : i32
! CHECK:              hlfir.assign %[[OUT_VAL]] to %[[RES_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:              omp.yield
! CHECK:            }
! CHECK:          }
! CHECK:          return
! CHECK:        }

subroutine omp_taskloop
  integer :: res, i
  !$omp taskloop
  do i = 1, 10
     res = res + 1
  end do
  !$omp end taskloop
end subroutine omp_taskloop


! CHECK-LABEL:  func.func @_QPomp_taskloop_private
! CHECK:           %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_taskloop_privateEi"}
! CHECK:           %[[DECL_I:.*]]:2 = hlfir.declare %[[ALLOCA_I]] {uniq_name = "_QFomp_taskloop_privateEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[ALLOCA_RES:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFomp_taskloop_privateEres"}
! CHECK:           %[[DECL_RES:.*]]:2 = hlfir.declare %[[ALLOCA_RES]] {uniq_name = "_QFomp_taskloop_privateEres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
subroutine omp_taskloop_private
  integer :: res, i
! CHECK:           omp.taskloop private(@[[RES_PRIVATE_TEST2]] %[[DECL_RES]]#0 -> %[[ARG0:.*]], @[[I_PRIVATE_TEST2]] %[[DECL_I]]#0 -> %[[ARG1:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:             omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
! CHECK:               %[[VAL1:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFomp_taskloop_privateEres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  !$omp taskloop private(res)
  do i = 1, 10
! CHECK:               %[[LOAD_RES:.*]] = fir.load %[[VAL1]]#0 : !fir.ref<i32>
! CHECK:               %[[C1_I32_1:.*]] = arith.constant 1 : i32
! CHECK:               %[[ADD_VAL:.*]] = arith.addi %[[LOAD_RES]], %[[C1_I32_1]] : i32
! CHECK:               hlfir.assign %[[ADD_VAL]] to %[[VAL1]]#0 : i32, !fir.ref<i32>
     res = res + 1
  end do
! CHECK:           return
! CHECK:         }
  !$omp end taskloop
end subroutine omp_taskloop_private

!===============================================================================
! `allocate` clause
!===============================================================================

! CHECK-LABEL:  func.func @_QPtaskloop_allocate
! CHECK:           %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtaskloop_allocateEi"}
! CHECK:           %[[DECL_I:.*]]:2 = hlfir.declare %[[ALLOCA_I]] {uniq_name = "_QFtaskloop_allocateEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[ALLOCA_X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtaskloop_allocateEx"}
! CHECK:           %[[DECL_X:.*]]:2 = hlfir.declare %[[ALLOCA_X]] {uniq_name = "_QFtaskloop_allocateEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
subroutine taskloop_allocate()
   use omp_lib
   integer :: x
   ! CHECK:           omp.taskloop allocate(%{{.*}} : i64 -> %[[DECL_X]]#0 : !fir.ref<i32>) 
   ! CHECK-SAME:      private(@[[X_PRIVATE_TEST_ALLOCATE]] %[[DECL_X]]#0 -> %[[ARG0:.*]], @[[I_PRIVATE_TEST_ALLOCATE]] %[[DECL_I]]#0 -> %[[ARG1:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
   !$omp taskloop allocate(omp_high_bw_mem_alloc: x) private(x)
   do i = 1, 100
      ! CHECK: arith.addi
      x = x + 12
      ! CHECK: omp.yield
   end do
   !$omp end taskloop
end subroutine taskloop_allocate

!===============================================================================
! `final` clause
!===============================================================================

! CHECK-LABEL:  func.func @_QPtaskloop_final
! CHECK:           %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtaskloop_finalEi"}
! CHECK:           %[[DECL_I:.*]]:2 = hlfir.declare %[[ALLOCA_I]] {uniq_name = "_QFtaskloop_finalEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
subroutine taskloop_final()
    ! CHECK:  omp.taskloop final(%true) private(@[[I_PRIVATE_FINAL]] %[[DECL_I]]#0 -> %[[ARG0:.*]] : !fir.ref<i32>) {
   !$omp taskloop final(.true.)
   do i = 1, 100
      ! CHECK: fir.call @_QPfoo()
      call foo()
   end do
   !$omp end taskloop
end subroutine

!===============================================================================
! `if` clause
!===============================================================================

! CHECK-LABEL:  func.func @_QPomp_taskloop_if
! CHECK:            %[[DECL_BAR:.*]]:2 = hlfir.declare %[[ARG0:.*]] dummy_scope %{{.*}}
! CHECK:           %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_taskloop_ifEi"}
! CHECK:           %[[DECL_I:.*]]:2 = hlfir.declare %[[ALLOCA_I]] {uniq_name = "_QFomp_taskloop_ifEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[LOAD_VAL:.*]] = fir.load %[[DECL_BAR]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_BAR:.*]] = fir.convert %[[LOAD_VAL]] : (!fir.logical<4>) -> i1
subroutine omp_taskloop_if(bar)
   logical, intent(inout) :: bar
   !CHECK: omp.taskloop if(%[[VAL_BAR]]) private(@[[I_PRIVATE_IF_TEST1]] %[[DECL_I]]#0 -> %[[ARG1:.*]] : !fir.ref<i32>) {
   !$omp taskloop if(bar)
   do i = 1, 10
      call foo()
   end do
   !$omp end taskloop
end subroutine omp_taskloop_if

!===============================================================================
! `mergeable` clause
!===============================================================================

! CHECK-LABEL:  func.func @_QPtest_mergeable
subroutine test_mergeable
  ! CHECK: omp.taskloop mergeable
  !$omp taskloop mergeable
  do i = 1, 10
  end do
  !$omp end taskloop
end subroutine test_mergeable

!===============================================================================
! `priority` clause
!===============================================================================

! CHECK-LABEL:  func.func @_QPtest_priority
! CHECK:          %[[VAL1:.*]]:2 = hlfir.declare %[[ARG0:.*]] dummy_scope %{{.*}}
! CHECK:          %[[LOAD_VAL:.*]] = fir.load %[[VAL1]]#0 : !fir.ref<i32>
subroutine test_priority(n)
   integer, intent(inout) :: n
   ! CHECK:  omp.taskloop priority(%[[LOAD_VAL]] : i32)
   !$omp taskloop priority(n)
   do i = 1, 10
   end do
   !$omp end taskloop
end subroutine test_priority

!===============================================================================
! `untied` clause
!===============================================================================

! CHECK-LABEL:  func.func @_QPomp_taskloop_untied
subroutine omp_taskloop_untied()
  ! CHECK: omp.taskloop untied
  !$omp taskloop untied
  do i = 1, 10
    call foo()
  end do
  !$omp end taskloop
end subroutine

!===============================================================================
! `nogroup` clause
!===============================================================================

subroutine omp_taskloop_nogroup()
  ! CHECK: omp.taskloop nogroup
  !$omp taskloop nogroup
  do i = 1, 10
    call foo()
  end do
  !$omp end taskloop
end subroutine

!===============================================================================
! `lastprivate` clause
!===============================================================================

! CHECK-LABEL:  func.func @_QPomp_taskloop_lastprivate
! CHECK:          %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_taskloop_lastprivateEi"}
! CHECK:          %[[DECL_I:.*]]:2 = hlfir.declare %[[ALLOCA_I]] {uniq_name = "_QFomp_taskloop_lastprivateEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          %[[ALLOCA_X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFomp_taskloop_lastprivateEx"}
! CHECK:          %[[DECL_X:.*]]:2 = hlfir.declare %[[ALLOCA_X]] {uniq_name = "_QFomp_taskloop_lastprivateEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
subroutine omp_taskloop_lastprivate()
   integer x
   x = 0
   ! CHECK:  omp.taskloop private(@[[LAST_PRIVATE_X]] %[[DECL_X]]#0 -> %[[ARG0]], @[[LAST_PRIVATE_I]] %[[DECL_I]]#0 -> %[[ARG1]] : !fir.ref<i32>, !fir.ref<i32>) {
   !$omp taskloop lastprivate(x)
   do i = 1, 100
      ! CHECK: %[[DECL_ARG0:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFomp_taskloop_lastprivateEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      ! CHECK: %[[LOAD_ARG0:.*]] = fir.load %[[DECL_ARG0]]#0 : !fir.ref<i32>
      ! CHECK: %[[RES_ADD:.*]] = arith.addi %[[LOAD_ARG0]], %{{.*}} : i32
      ! CHECK:  hlfir.assign %[[RES_ADD]] to %[[DECL_ARG0]]#0 : i32, !fir.ref<i32>
      x = x + 1
      ! CHECK:  %[[SELCT_RESULT:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : i1
      ! CHECK:  fir.if %[[SELCT_RESULT]] {
      ! CHECK:    %[[LOADED_SUM:.*]] = fir.load %[[DECL_ARG0]]#0 : !fir.ref<i32>
      ! CHECK:    hlfir.assign %[[LOADED_SUM]] to %[[DECL_X]]#0 : i32, !fir.ref<i32>
      ! CHECK:  }
      ! CHECK:  omp.yield
   end do
   !$omp end taskloop
end subroutine omp_taskloop_lastprivate
