! RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

program reduce_pointer
  integer, pointer :: v
  integer i

  allocate(v)
  v = 0

  !$omp parallel do private(i) reduction(+:v)
  do i = 1, 5
    v = v + 42
  end do
  !$omp end parallel do

  print *,v
  deallocate(v)
end program

! CHECK-LABEL:   omp.declare_reduction @add_reduction_byref_box_ptr_i32 : !fir.ref<!fir.box<!fir.ptr<i32>>> alloc {
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<i32>>
! CHECK:           omp.yield(%[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK-LABEL:   } init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>, %[[VAL_3:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>):
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ptr<i32>) -> i64
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:           fir.if %[[VAL_7]] {
! CHECK:             %[[VAL_8:.*]] = fir.embox %[[VAL_4]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:             fir.store %[[VAL_8]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           } else {
! CHECK:             %[[VAL_9:.*]] = fir.allocmem i32
! CHECK:             fir.store %[[VAL_1]] to %[[VAL_9]] : !fir.heap<i32>
! CHECK:             %[[VAL_10:.*]] = fir.embox %[[VAL_9]] : (!fir.heap<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:             fir.store %[[VAL_10]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           }
! CHECK:           omp.yield(%[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>):
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:           %[[VAL_5:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_4]] : !fir.ptr<i32>
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_5]] : !fir.ptr<i32>
! CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_6]], %[[VAL_7]] : i32
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_4]] : !fir.ptr<i32>
! CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK-LABEL:   }  cleanup {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>):
! CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ptr<i32>) -> i64
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:           fir.if %[[VAL_5]] {
! CHECK:             %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (!fir.ptr<i32>) -> !fir.heap<i32>
! CHECK:             fir.freemem %[[VAL_6]] : !fir.heap<i32>
! CHECK:           }
! CHECK:           omp.yield
! CHECK:         }

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "reduce_pointer"} {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "v", uniq_name = "_QFEv"}
! CHECK:           %[[VAL_3:.*]] = fir.zero_bits !fir.ptr<i32>
! CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_3]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = {{.*}}<pointer>, uniq_name = "_QFEv"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:           %[[VAL_6:.*]] = arith.constant false
! CHECK:           %[[VAL_7:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_8:.*]] = fir.address_of(
! CHECK:           %[[VAL_9:.*]] = arith.constant 8 : i32
! CHECK:           %[[VAL_10:.*]] = fir.zero_bits !fir.ptr<i32>
! CHECK:           %[[VAL_11:.*]] = fir.embox %[[VAL_10]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:           fir.store %[[VAL_11]] to %[[VAL_5]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.char<{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_14:.*]] = fir.call @_FortranAPointerAllocate(%[[VAL_12]], %[[VAL_6]], %[[VAL_7]], %[[VAL_13]], %[[VAL_9]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_15:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_17:.*]] = fir.box_addr %[[VAL_16]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:           hlfir.assign %[[VAL_15]] to %[[VAL_17]] : i32, !fir.ptr<i32>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_18:.*]] = fir.alloca i32 {bindc_name = "i", pinned, uniq_name = "_QFEi"}
! CHECK:             %[[VAL_19:.*]]:2 = hlfir.declare %[[VAL_18]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[VAL_20:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_21:.*]] = arith.constant 5 : i32
! CHECK:             %[[VAL_22:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop reduction(byref @add_reduction_byref_box_ptr_i32 %[[VAL_5]]#0 -> %[[VAL_23:.*]] : !fir.ref<!fir.box<!fir.ptr<i32>>>) {
! CHECK:               omp.loop_nest (%[[VAL_24:.*]]) : i32 = (%[[VAL_20]]) to (%[[VAL_21]]) inclusive step (%[[VAL_22]]) {
! CHECK:                 %[[VAL_25:.*]]:2 = hlfir.declare %[[VAL_23]] {fortran_attrs = {{.*}}<pointer>, uniq_name = "_QFEv"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:                 fir.store %[[VAL_24]] to %[[VAL_19]]#1 : !fir.ref<i32>
! CHECK:                 %[[VAL_26:.*]] = fir.load %[[VAL_25]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:                 %[[VAL_27:.*]] = fir.box_addr %[[VAL_26]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:                 %[[VAL_28:.*]] = fir.load %[[VAL_27]] : !fir.ptr<i32>
! CHECK:                 %[[VAL_29:.*]] = arith.constant 42 : i32
! CHECK:                 %[[VAL_30:.*]] = arith.addi %[[VAL_28]], %[[VAL_29]] : i32
! CHECK:                 %[[VAL_31:.*]] = fir.load %[[VAL_25]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:                 %[[VAL_32:.*]] = fir.box_addr %[[VAL_31]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:                 hlfir.assign %[[VAL_30]] to %[[VAL_32]] : i32, !fir.ptr<i32>
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
