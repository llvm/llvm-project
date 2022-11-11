! Test that temps are always created of parenthesized arguments in
! calls.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPfoo_num_scalar(
! CHECK-SAME:      %[[VAL_0:.*]]: !fir.ref<i32>{{.*}}) {
subroutine foo_num_scalar(x)
  integer :: x
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32
  call bar_num_scalar(x)
! CHECK:         fir.call @_QPbar_num_scalar(%[[VAL_0]]) : (!fir.ref<i32>) -> ()
  call bar_num_scalar((x))
! CHECK:         %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_3:.*]] = fir.no_reassoc %[[VAL_2]] : i32
! CHECK:         fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:         fir.call @_QPbar_num_scalar(%[[VAL_1]]) : (!fir.ref<i32>) -> ()
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPfoo_char_scalar(
! CHECK-SAME:         %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine foo_char_scalar(x)
  character(5) :: x
! CHECK:         %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_2:.*]] = arith.constant 5 : index
! CHECK:         %[[VAL_3:.*]] = fir.emboxchar %[[VAL_1]]#0, %[[VAL_2]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:         fir.call @_QPbar_char_scalar(%[[VAL_3]]) : (!fir.boxchar<1>) -> ()
  call bar_char_scalar(x)
! CHECK:         %[[VAL_4:.*]] = fir.no_reassoc %[[VAL_1]]#0 : !fir.ref<!fir.char<1,?>>
! CHECK:         %[[VAL_5:.*]] = fir.alloca !fir.char<1,5> {bindc_name = ".chrtmp"}
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_2]] : (index) -> i64
! CHECK:         %[[VAL_8:.*]] = arith.muli %[[VAL_6]], %[[VAL_7]] : i64
! CHECK:         %[[VAL_9:.*]] = arith.constant false
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:         fir.call @llvm.memmove.p0.p0.i64(%[[VAL_10]], %[[VAL_11]], %[[VAL_8]], %[[VAL_9]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:         %[[VAL_13:.*]] = fir.emboxchar %[[VAL_12]], %[[VAL_2]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:         fir.call @_QPbar_char_scalar(%[[VAL_13]]) : (!fir.boxchar<1>) -> ()
! CHECK:         return
! CHECK:       }
  call bar_char_scalar((x))
end subroutine

! CHECK-LABEL: func @_QPfoo_num_array(
! CHECK-SAME:                         %[[VAL_0:.*]]: !fir.ref<!fir.array<100xi32>>{{.*}}) {
subroutine foo_num_array(x)
  integer :: x(100)
  call bar_num_array(x)
! CHECK:         %[[VAL_1:.*]] = arith.constant 100 : index
! CHECK:         fir.call @_QPbar_num_array(%[[VAL_0]]) : (!fir.ref<!fir.array<100xi32>>) -> ()
  call bar_num_array((x))
! CHECK:         %[[VAL_3:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_4:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_5:.*]] = fir.array_load %[[VAL_0]](%[[VAL_4]]) : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.array<100xi32>
! CHECK:         %[[VAL_6:.*]] = fir.allocmem !fir.array<100xi32>
! CHECK:         %[[VAL_7:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_6]](%[[VAL_7]]) : (!fir.heap<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.array<100xi32>
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_11:.*]] = arith.subi %[[VAL_3]], %[[VAL_9]] : index
! CHECK:         %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_9]] unordered iter_args(%[[VAL_14:.*]] = %[[VAL_8]]) -> (!fir.array<100xi32>) {
! CHECK:           %[[VAL_15:.*]] = fir.array_fetch %[[VAL_5]], %[[VAL_13]] : (!fir.array<100xi32>, index) -> i32
! CHECK:           %[[VAL_16:.*]] = fir.no_reassoc %[[VAL_15]] : i32
! CHECK:           %[[VAL_17:.*]] = fir.array_update %[[VAL_14]], %[[VAL_16]], %[[VAL_13]] : (!fir.array<100xi32>, i32, index) -> !fir.array<100xi32>
! CHECK:           fir.result %[[VAL_17]] : !fir.array<100xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_8]], %[[VAL_18:.*]] to %[[VAL_6]] : !fir.array<100xi32>, !fir.array<100xi32>, !fir.heap<!fir.array<100xi32>>
! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_6]] : (!fir.heap<!fir.array<100xi32>>) -> !fir.ref<!fir.array<100xi32>>
! CHECK:         fir.call @_QPbar_num_array(%[[VAL_19]]) : (!fir.ref<!fir.array<100xi32>>) -> ()
! CHECK:         fir.freemem %[[VAL_6]] : !fir.heap<!fir.array<100xi32>>
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPfoo_char_array(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine foo_char_array(x)
  ! CHECK: %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_2:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_3:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,10>>>
  ! CHECK: %[[VAL_4:.*]] = arith.constant 100 : index
  ! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.array<100x!fir.char<1,10>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_6:.*]] = fir.emboxchar %[[VAL_5]], %[[VAL_2]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPbar_char_array(%[[VAL_6]]) : (!fir.boxchar<1>) -> ()
  ! CHECK: %[[VAL_8:.*]] = arith.constant 100 : index
  ! CHECK: %[[VAL_9:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_10:.*]] = fir.array_load %[[VAL_3]](%[[VAL_9]]) : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<100x!fir.char<1,10>>
  ! CHECK: %[[VAL_11:.*]] = fir.allocmem !fir.array<100x!fir.char<1,10>>
  ! CHECK: %[[VAL_12:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_13:.*]] = fir.array_load %[[VAL_11]](%[[VAL_12]]) : (!fir.heap<!fir.array<100x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<100x!fir.char<1,10>>
  ! CHECK: %[[VAL_14:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_15:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_16:.*]] = arith.subi %[[VAL_8]], %[[VAL_14]] : index
  ! CHECK: %[[VAL_17:.*]] = fir.do_loop %[[VAL_18:.*]] = %[[VAL_15]] to %[[VAL_16]] step %[[VAL_14]] unordered iter_args(%[[VAL_19:.*]] = %[[VAL_13]]) -> (!fir.array<100x!fir.char<1,10>>) {
  ! CHECK: %[[VAL_20:.*]] = fir.array_access %[[VAL_10]], %[[VAL_18]] : (!fir.array<100x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_21:.*]] = fir.no_reassoc %[[VAL_20]] : !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_22:.*]] = fir.array_access %[[VAL_19]], %[[VAL_18]] : (!fir.array<100x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_23:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_24:.*]] = arith.constant 1 : i64
  ! CHECK: %[[VAL_25:.*]] = fir.convert %[[VAL_23]] : (index) -> i64
  ! CHECK: %[[VAL_26:.*]] = arith.muli %[[VAL_24]], %[[VAL_25]] : i64
  ! CHECK: %[[VAL_27:.*]] = arith.constant false
  ! CHECK: %[[VAL_28:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_29:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[VAL_28]], %[[VAL_29]], %[[VAL_26]], %[[VAL_27]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_30:.*]] = fir.array_amend %[[VAL_19]], %[[VAL_22]] : (!fir.array<100x!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>) -> !fir.array<100x!fir.char<1,10>>
  ! CHECK: fir.result %[[VAL_30]] : !fir.array<100x!fir.char<1,10>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_13]], %[[VAL_31:.*]] to %[[VAL_11]] : !fir.array<100x!fir.char<1,10>>, !fir.array<100x!fir.char<1,10>>, !fir.heap<!fir.array<100x!fir.char<1,10>>>
  ! CHECK: %[[VAL_32:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_33:.*]] = fir.convert %[[VAL_11]] : (!fir.heap<!fir.array<100x!fir.char<1,10>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_34:.*]] = fir.emboxchar %[[VAL_33]], %[[VAL_32]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPbar_char_array(%[[VAL_34]]) : (!fir.boxchar<1>) -> ()
  ! CHECK: fir.freemem %[[VAL_11]] : !fir.heap<!fir.array<100x!fir.char<1,10>>>

  character(10) :: x(100)
  call bar_char_array(x)
  call bar_char_array((x))
  ! CHECK:         return
  ! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPfoo_num_array_box(
! CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<!fir.array<100xi32>>{{.*}}) {
subroutine foo_num_array_box(x)
  ! CHECK: %[[VAL_1:.*]] = arith.constant 100 : index
  ! CHECK: %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_3:.*]] = fir.embox %[[VAL_0]](%[[VAL_2]]) : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<100xi32>>
  ! CHECK: %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.box<!fir.array<100xi32>>) -> !fir.box<!fir.array<?xi32>>
  ! CHECK: fir.call @_QPbar_num_array_box(%[[VAL_4]]) : (!fir.box<!fir.array<?xi32>>) -> ()
  ! CHECK: %[[VAL_6:.*]] = arith.constant 100 : index
  ! CHECK: %[[VAL_7:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_8:.*]] = fir.array_load %[[VAL_0]](%[[VAL_7]]) : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.array<100xi32>
  ! CHECK: %[[VAL_9:.*]] = fir.allocmem !fir.array<100xi32>
  ! CHECK: %[[VAL_10:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_11:.*]] = fir.array_load %[[VAL_9]](%[[VAL_10]]) : (!fir.heap<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.array<100xi32>
  ! CHECK: %[[VAL_12:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_13:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_14:.*]] = arith.subi %[[VAL_6]], %[[VAL_12]] : index
  ! CHECK: %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_13]] to %[[VAL_14]] step %[[VAL_12]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_11]]) -> (!fir.array<100xi32>) {
  ! CHECK: %[[VAL_18:.*]] = fir.array_fetch %[[VAL_8]], %[[VAL_16]] : (!fir.array<100xi32>, index) -> i32
  ! CHECK: %[[VAL_19:.*]] = fir.no_reassoc %[[VAL_18]] : i32
  ! CHECK: %[[VAL_20:.*]] = fir.array_update %[[VAL_17]], %[[VAL_19]], %[[VAL_16]] : (!fir.array<100xi32>, i32, index) -> !fir.array<100xi32>
  ! CHECK: fir.result %[[VAL_20]] : !fir.array<100xi32>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_11]], %[[VAL_21:.*]] to %[[VAL_9]] : !fir.array<100xi32>, !fir.array<100xi32>, !fir.heap<!fir.array<100xi32>>
  ! CHECK: %[[VAL_22:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_23:.*]] = fir.embox %[[VAL_9]](%[[VAL_22]]) : (!fir.heap<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<100xi32>>
  ! CHECK: %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (!fir.box<!fir.array<100xi32>>) -> !fir.box<!fir.array<?xi32>>
  ! CHECK: fir.call @_QPbar_num_array_box(%[[VAL_24]]) : (!fir.box<!fir.array<?xi32>>) -> ()
  ! CHECK: fir.freemem %[[VAL_9]] : !fir.heap<!fir.array<100xi32>>

  integer :: x(100)
  interface
   subroutine bar_num_array_box(x)
     integer :: x(:)
   end subroutine
  end interface
  call bar_num_array_box(x)
  call bar_num_array_box((x))
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPfoo_char_array_box(
! CHECK-SAME:          %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}) {
subroutine foo_char_array_box(x, n)
  ! CHECK: %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,10>>>
  ! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> i64
  ! CHECK: %[[VAL_6A:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
  ! CHECK: %[[C0:.*]] = arith.constant 0 : index 
  ! CHECK: %[[CMP:.*]] = arith.cmpi sgt, %[[VAL_6A]], %[[C0]] : index 
  ! CHECK: %[[VAL_6:.*]] = arith.select %[[CMP]], %[[VAL_6A]], %[[C0]] : index
  ! CHECK: %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_8:.*]] = fir.embox %[[VAL_3]](%[[VAL_7]]) : (!fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.char<1,10>>>
  ! CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: fir.call @_QPbar_char_array_box(%[[VAL_9]]) : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> ()
  ! CHECK: %[[VAL_10:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_11:.*]] = fir.array_load %[[VAL_3]](%[[VAL_10]]) : (!fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<?x!fir.char<1,10>>
  ! CHECK: %[[VAL_12:.*]] = fir.allocmem !fir.array<?x!fir.char<1,10>>, %[[VAL_6]] {uniq_name = ".array.expr"}
  ! CHECK: %[[VAL_13:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_14:.*]] = fir.array_load %[[VAL_12]](%[[VAL_13]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<?x!fir.char<1,10>>
  ! CHECK: %[[VAL_15:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_16:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_17:.*]] = arith.subi %[[VAL_6]], %[[VAL_15]] : index
  ! CHECK: %[[VAL_18:.*]] = fir.do_loop %[[VAL_19:.*]] = %[[VAL_16]] to %[[VAL_17]] step %[[VAL_15]] unordered iter_args(%[[VAL_20:.*]] = %[[VAL_14]]) -> (!fir.array<?x!fir.char<1,10>>) {
  ! CHECK: %[[VAL_21:.*]] = fir.array_access %[[VAL_11]], %[[VAL_19]] : (!fir.array<?x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_22:.*]] = fir.no_reassoc %[[VAL_21]] : !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_23:.*]] = fir.array_access %[[VAL_20]], %[[VAL_19]] : (!fir.array<?x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_24:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_25:.*]] = arith.constant 1 : i64
  ! CHECK: %[[VAL_26:.*]] = fir.convert %[[VAL_24]] : (index) -> i64
  ! CHECK: %[[VAL_27:.*]] = arith.muli %[[VAL_25]], %[[VAL_26]] : i64
  ! CHECK: %[[VAL_28:.*]] = arith.constant false
  ! CHECK: %[[VAL_29:.*]] = fir.convert %[[VAL_23]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_30:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[VAL_29]], %[[VAL_30]], %[[VAL_27]], %[[VAL_28]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_31:.*]] = fir.array_amend %[[VAL_20]], %[[VAL_23]] : (!fir.array<?x!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>) -> !fir.array<?x!fir.char<1,10>>
  ! CHECK: fir.result %[[VAL_31]] : !fir.array<?x!fir.char<1,10>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_14]], %[[VAL_32:.*]] to %[[VAL_12]] : !fir.array<?x!fir.char<1,10>>, !fir.array<?x!fir.char<1,10>>, !fir.heap<!fir.array<?x!fir.char<1,10>>>
  ! CHECK: %[[VAL_33:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_34:.*]] = fir.embox %[[VAL_12]](%[[VAL_33]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.char<1,10>>>
  ! CHECK: %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: fir.call @_QPbar_char_array_box(%[[VAL_35]]) : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> ()
  ! CHECK: fir.freemem %[[VAL_12]] : !fir.heap<!fir.array<?x!fir.char<1,10>>>

  integer :: n
  character(10) :: x(n)
  interface
   subroutine bar_char_array_box(x)
     character(*) :: x(:)
   end subroutine
  end interface
  call bar_char_array_box(x)
  call bar_char_array_box((x))
  ! CHECK:         return
  ! CHECK:       }
end subroutine
