! Test lowering of references to pointers
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Assigning/reading to scalar pointer target.
! CHECK-LABEL: func @_QPscal_ptr(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>{{.*}})
subroutine scal_ptr(p)
  real, pointer :: p
  real :: x
  ! CHECK: %[[boxload:.*]] = fir.load %[[arg0]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[boxload]]
  ! CHECK: fir.store %{{.*}} to %[[addr]]
  p = 3.

  ! CHECK: %[[boxload2:.*]] = fir.load %[[arg0]]
  ! CHECK: %[[addr2:.*]] = fir.box_addr %[[boxload2]]
  ! CHECK: %[[val:.*]] = fir.load %[[addr2]]
  ! CHECK: fir.store %[[val]] to %{{.*}}
  x = p
end subroutine

! Assigning/reading scalar character pointer target.
! CHECK-LABEL: func @_QPchar_ptr(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,12>>>>{{.*}})
subroutine char_ptr(p)
  character(12), pointer :: p
  character(12) :: x

  ! CHECK-DAG: %[[str:.*]] = fir.address_of(@_QQcl.68656C6C6F20776F726C6421) : !fir.ref<!fir.char<1,12>>
  ! CHECK: %[[boxload:.*]] = fir.load %[[arg0]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[boxload]]
  ! CHECK-DAG: %[[one:.*]] = arith.constant 1
  ! CHECK-DAG: %[[size:.*]] = fir.convert %{{.*}} : (index) -> i64
  ! CHECK: %[[count:.*]] = arith.muli %[[one]], %[[size]] : i64
  ! CHECK: %[[dst:.*]] = fir.convert %[[addr]] : (!fir.ptr<!fir.char<1,12>>) -> !fir.ref<i8>
  ! CHECK: %[[src:.*]] = fir.convert %[[str]] : (!fir.ref<!fir.char<1,12>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[dst]], %[[src]], %5, %false) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  p = "hello world!"

  ! CHECK: %[[boxload2:.*]] = fir.load %[[arg0]]
  ! CHECK: %[[addr2:.*]] = fir.box_addr %[[boxload2]]
  ! CHECK: %[[count:.*]] = arith.muli %{{.*}}, %{{.*}} : i64
  ! CHECK: %[[dst:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.char<1,12>>) -> !fir.ref<i8>
  ! CHECK: %[[src:.*]] = fir.convert %[[addr2]] : (!fir.ptr<!fir.char<1,12>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[dst]], %[[src]], %[[count]], %{{.*}}) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  x = p
end subroutine

! Reading from pointer in array expression
! CHECK-LABEL: func @_QParr_ptr_read(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}})
subroutine arr_ptr_read(p)
  real, pointer :: p(:)
  real :: x(100)
  ! CHECK: %[[boxload:.*]] = fir.load %[[arg0]]
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %[[boxload]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK: %[[lb:.*]] = fir.shift %[[dims]]#0 : (index) -> !fir.shift<1>
  ! CHECK: fir.array_load %[[boxload]](%[[lb]]) : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>) -> !fir.array<?xf32>
  x = p
end subroutine

! Reading from contiguous pointer in array expression
! CHECK-LABEL: func @_QParr_contig_ptr_read(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {{{.*}}, fir.contiguous})
subroutine arr_contig_ptr_read(p)
  real, pointer, contiguous :: p(:)
  real :: x(100)
  ! CHECK: %[[boxload:.*]] = fir.load %[[arg0]]
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[boxload]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[boxload]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
  ! CHECK-DAG: %[[shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: fir.array_load %[[addr]](%[[shape]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.array<?xf32>
  x = p
end subroutine

! Assigning to pointer target in array expression

  ! CHECK-LABEL: func @_QParr_ptr_target_write(
  ! CHECK-SAME:                                %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}}) {
  ! CHECK:         %[[VAL_1:.*]] = arith.constant 100 : index
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.array<100xf32> {bindc_name = "x", uniq_name = "_QFarr_ptr_target_writeEx"}
  ! CHECK:         %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:         %[[VAL_4:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_6:.*]] = arith.constant 2 : i64
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
  ! CHECK:         %[[VAL_8:.*]] = arith.constant 6 : i64
  ! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
  ! CHECK:         %[[VAL_10:.*]] = arith.constant 601 : i64
  ! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
  ! CHECK:         %[[VAL_12:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_13:.*]] = arith.subi %[[VAL_11]], %[[VAL_7]] : index
  ! CHECK:         %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_9]] : index
  ! CHECK:         %[[VAL_15:.*]] = arith.divsi %[[VAL_14]], %[[VAL_9]] : index
  ! CHECK:         %[[VAL_16:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_12]] : index
  ! CHECK:         %[[VAL_17:.*]] = arith.select %[[VAL_16]], %[[VAL_15]], %[[VAL_12]] : index
  ! CHECK:         %[[VAL_18:.*]] = fir.shift %[[VAL_5]]#0 : (index) -> !fir.shift<1>
  ! CHECK:         %[[VAL_19:.*]] = fir.slice %[[VAL_7]], %[[VAL_11]], %[[VAL_9]] : (index, index, index) -> !fir.slice<1>
  ! CHECK:         %[[VAL_20:.*]] = fir.array_load %[[VAL_3]](%[[VAL_18]]) {{\[}}%[[VAL_19]]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, !fir.slice<1>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_21:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_22:.*]] = fir.array_load %[[VAL_2]](%[[VAL_21]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  ! CHECK:         %[[VAL_23:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_24:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_25:.*]] = arith.subi %[[VAL_17]], %[[VAL_23]] : index
  ! CHECK:         %[[VAL_26:.*]] = fir.do_loop %[[VAL_27:.*]] = %[[VAL_24]] to %[[VAL_25]] step %[[VAL_23]] unordered iter_args(%[[VAL_28:.*]] = %[[VAL_20]]) -> (!fir.array<?xf32>) {
  ! CHECK:           %[[VAL_29:.*]] = fir.array_fetch %[[VAL_22]], %[[VAL_27]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK:           %[[VAL_30:.*]] = fir.array_update %[[VAL_28]], %[[VAL_29]], %[[VAL_27]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
  ! CHECK:           fir.result %[[VAL_30]] : !fir.array<?xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_20]], %[[VAL_31:.*]] to %[[VAL_3]]{{\[}}%[[VAL_19]]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.slice<1>
  ! CHECK:         return
  ! CHECK:       }

subroutine arr_ptr_target_write(p)
  real, pointer :: p(:)
  real :: x(100)
  p(2:601:6) = x
end subroutine

! Assigning to contiguous pointer target in array expression

  ! CHECK-LABEL: func @_QParr_contig_ptr_target_write(
  ! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {{{.*}}, fir.contiguous}) {
  ! CHECK:         %[[VAL_1:.*]] = arith.constant 100 : index
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.array<100xf32> {bindc_name = "x", uniq_name = "_QFarr_contig_ptr_target_writeEx"}
  ! CHECK:         %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:         %[[VAL_4:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_6:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
  ! CHECK:         %[[VAL_7:.*]] = arith.constant 2 : i64
  ! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
  ! CHECK:         %[[VAL_9:.*]] = arith.constant 6 : i64
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
  ! CHECK:         %[[VAL_11:.*]] = arith.constant 601 : i64
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
  ! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_14:.*]] = arith.subi %[[VAL_12]], %[[VAL_8]] : index
  ! CHECK:         %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_10]] : index
  ! CHECK:         %[[VAL_16:.*]] = arith.divsi %[[VAL_15]], %[[VAL_10]] : index
  ! CHECK:         %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_13]] : index
  ! CHECK:         %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_16]], %[[VAL_13]] : index
  ! CHECK:         %[[VAL_19:.*]] = fir.shape_shift %[[VAL_5]]#0, %[[VAL_5]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK:         %[[VAL_20:.*]] = fir.slice %[[VAL_8]], %[[VAL_12]], %[[VAL_10]] : (index, index, index) -> !fir.slice<1>
  ! CHECK:         %[[VAL_21:.*]] = fir.array_load %[[VAL_6]](%[[VAL_19]]) {{\[}}%[[VAL_20]]] : (!fir.ptr<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_22:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_23:.*]] = fir.array_load %[[VAL_2]](%[[VAL_22]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  ! CHECK:         %[[VAL_24:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_25:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_26:.*]] = arith.subi %[[VAL_18]], %[[VAL_24]] : index
  ! CHECK:         %[[VAL_27:.*]] = fir.do_loop %[[VAL_28:.*]] = %[[VAL_25]] to %[[VAL_26]] step %[[VAL_24]] unordered iter_args(%[[VAL_29:.*]] = %[[VAL_21]]) -> (!fir.array<?xf32>) {
  ! CHECK:           %[[VAL_30:.*]] = fir.array_fetch %[[VAL_23]], %[[VAL_28]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK:           %[[VAL_31:.*]] = fir.array_update %[[VAL_29]], %[[VAL_30]], %[[VAL_28]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
  ! CHECK:           fir.result %[[VAL_31]] : !fir.array<?xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_21]], %[[VAL_32:.*]] to %[[VAL_6]]{{\[}}%[[VAL_20]]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.ptr<!fir.array<?xf32>>, !fir.slice<1>
  ! CHECK:         return
  ! CHECK:       }

subroutine arr_contig_ptr_target_write(p)
  real, pointer, contiguous :: p(:)
  real :: x(100)
  p(2:601:6) = x
end subroutine

! CHECK-LABEL: func @_QPpointer_result_as_value
subroutine pointer_result_as_value()
  ! Test that function pointer results used as values are correctly loaded.
  interface
    function returns_int_pointer()
      integer, pointer :: returns_int_pointer
    end function
  end interface
! CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = ".result"}
! CHECK:  %[[VAL_6:.*]] = fir.call @_QPreturns_int_pointer() : () -> !fir.box<!fir.ptr<i32>>
! CHECK:  fir.save_result %[[VAL_6]] to %[[VAL_0]] : !fir.box<!fir.ptr<i32>>, !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:  fir.load %[[VAL_8]] : !fir.ptr<i32>
  print *, returns_int_pointer()
end subroutine
