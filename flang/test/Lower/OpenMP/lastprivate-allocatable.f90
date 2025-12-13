! RUN: %flang_fc1 -emit-hlfir -o - -fopenmp %s | FileCheck %s
! RUN: bbc -emit-hlfir -o - -fopenmp %s | FileCheck %s

! CHECK-LABEL:   func.func @_QQmain()
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "a", uniq_name = "_QFEa"}
! CHECK:           %[[VAL_1:.*]] = fir.zero_bits !fir.heap<i32>
! CHECK:           %[[VAL_2:.*]] = fir.embox %[[VAL_1]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK:           fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:           omp.parallel {
! CHECK:             omp.wsloop private(@{{.*}} %{{.*}} -> %{{.*}}, @{{.*}} %{{.*}} -> %[[VAL_17:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<i32>) private_barrier {
! CHECK:               omp.loop_nest
! CHECK:                   %[[VAL_16:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:                   %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_17]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!                        [...]
!                        if this is the last iteration
! CHECK:                 fir.if %{{.*}} {
!                          store loop IV
! CHECK:                   hlfir.assign %{{.*}} to %[[VAL_18]]#0 : i32, !fir.ref<i32>
!                          assign private variable to original copy: realloc
! CHECK:                   %[[VAL_23:.*]] = fir.load %[[VAL_16]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:                   %[[VAL_24:.*]] = fir.box_addr %[[VAL_23]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:                   %[[VAL_25:.*]] = fir.load %[[VAL_24]] : !fir.heap<i32>
! CHECK:                   hlfir.assign %[[VAL_25]] to %[[VAL_3]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-NEXT:            }
! CHECK-NEXT:            omp.yield
! CHECK-NEXT:          }
program lastprivate_allocatable
  integer, allocatable :: a
  integer :: i
  ! a is unallocated here
  !$omp parallel do lastprivate(a)
  do i=1,1
    a = 42
  enddo
  !$omp end parallel do
  ! a should be allocated here
end program

! CHECK-LABEL:  func @_QPlastprivate_realloc()
! CHECK:          %[[A:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFlastprivate_reallocEa"} :
! CHECK-SAME:       (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>) ->
! CHECK-SAME:       (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>)
! CHECK:          omp.parallel {
! CHECK:            %[[A_PRIV:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFlastprivate_reallocEa"} :
! CHECK-SAME:         (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>) ->
! CHECK-SAME:         (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>)
! CHECK:            omp.sections {
! CHECK:              omp.section {
! CHECK:                fir.load
! CHECK:                %[[TEMP:.*]] = fir.load %[[A_PRIV:.*]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
! CHECK:                hlfir.assign %[[TEMP]] to %[[A]]#0 realloc : !fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>,
! CHECK-SAME:             !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
! CHECK:              }
! CHECK:            }
! CHECK:          }
subroutine lastprivate_realloc()
  complex, allocatable :: a(:)

  allocate(a(2))
  !$omp parallel
    !$omp sections lastprivate(a)
      !$omp section
        deallocate(a)
        allocate(a(3))
    !$omp end sections
  !$omp end parallel
end subroutine
