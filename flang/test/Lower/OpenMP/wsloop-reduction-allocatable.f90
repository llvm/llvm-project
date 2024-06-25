! RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

program reduce
integer :: i = 0
integer, allocatable :: r

allocate(r)
r = 0

!$omp parallel do reduction(+:r)
do i=0,10
  r = i
enddo
!$omp end parallel do

print *,r

end program

! CHECK:         omp.declare_reduction @add_reduction_byref_box_heap_i32 : !fir.ref<!fir.box<!fir.heap<i32>>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>):
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:           %[[LOAD:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
! CHECK:           %[[ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:           %[[ADDRI:.*]] = fir.convert %[[ADDR]] : (!fir.heap<i32>) -> i64
! CHECK:           %[[C0_I64:.*]] = arith.constant 0 : i64
! CHECK:           %[[IS_NULL:.*]] = arith.cmpi eq, %[[ADDRI]], %[[C0_I64]] : i64
! CHECK:           fir.if %[[IS_NULL]] {
! CHECK:             %[[NULL_BOX:.*]] = fir.embox %[[ADDR]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK:             fir.store %[[NULL_BOX]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<i32>>
! CHECK:           } else {
! CHECK:             %[[VAL_3:.*]] = fir.allocmem i32
! CHECK:             fir.store %[[VAL_1]] to %[[VAL_3]] : !fir.heap<i32>
! CHECK:             %[[VAL_4:.*]] = fir.embox %[[VAL_3]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK:             fir.store %[[VAL_4]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           }
! CHECK:           omp.yield(%[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:         } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>):
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:           %[[VAL_5:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_4]] : !fir.heap<i32>
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_5]] : !fir.heap<i32>
! CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_6]], %[[VAL_7]] : i32
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_4]] : !fir.heap<i32>
! CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:         }  cleanup {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>):
! CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.heap<i32>) -> i64
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:           fir.if %[[VAL_5]] {
! CHECK:             fir.freemem %[[VAL_2]] : !fir.heap<i32>
! CHECK:           }
! CHECK:           omp.yield
! CHECK:         }

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "reduce"} {
! CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QFEi) : !fir.ref<i32>
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "r", uniq_name = "_QFEr"}
! CHECK:           %[[VAL_3:.*]] = fir.zero_bits !fir.heap<i32>
! CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_3]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEr"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:           %[[VAL_6:.*]] = fir.allocmem i32 {fir.must_be_heap = true, uniq_name = "_QFEr.alloc"}
! CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_6]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_5]]#1 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[VAL_8:.*]] = arith.constant 0 : i32
! CHECK:           hlfir.assign %[[VAL_8]] to %[[VAL_5]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_9:.*]] = fir.alloca i32 {bindc_name = "i", pinned, {{.*}}}
! CHECK:             %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[VAL_11:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_12:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_13:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop reduction(byref @add_reduction_byref_box_heap_i32 %[[VAL_5]]#0 -> %[[VAL_14:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>>) {
! CHECK-NEXT:          omp.loop_nest (%[[VAL_15:.*]]) : i32 = (%[[VAL_11]]) to (%[[VAL_12]]) inclusive step (%[[VAL_13]]) {
! CHECK:                 %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_14]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEr"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:                 fir.store %[[VAL_15]] to %[[VAL_10]]#1 : !fir.ref<i32>
! CHECK:                 %[[VAL_17:.*]] = fir.load %[[VAL_10]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_18:.*]] = fir.load %[[VAL_16]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:                 %[[VAL_19:.*]] = fir.box_addr %[[VAL_18]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:                 hlfir.assign %[[VAL_17]] to %[[VAL_19]] : i32, !fir.heap<i32>
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }

