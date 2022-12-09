! Test passing pointer, allocatables, and optional assumed shapes to optional
! explicit shapes (see F2018 15.5.2.12).
! RUN: bbc -emit-fir %s -o - | FileCheck %s
module optional_tests
implicit none
interface
subroutine takes_opt_scalar(i)
  integer, optional :: i
end subroutine
subroutine takes_opt_scalar_char(c)
  character(*), optional :: c
end subroutine
subroutine takes_opt_explicit_shape(x)
  real, optional :: x(100)
end subroutine
subroutine takes_opt_explicit_shape_intentout(x)
  real, optional, intent(out) :: x(100)
end subroutine
subroutine takes_opt_explicit_shape_intentin(x)
  real, optional, intent(in) :: x(100)
end subroutine
subroutine takes_opt_explicit_shape_char(c)
  character(*), optional :: c(100)
end subroutine
function returns_pointer()
  real, pointer :: returns_pointer(:)
end function
end interface
contains

! -----------------------------------------------------------------------------
!     Test passing scalar pointers and allocatables to an optional
! -----------------------------------------------------------------------------
! Here, nothing optional specific is expected, the address is passed, and its
! allocation/association status match the dummy presence status.

! CHECK-LABEL: func @_QMoptional_testsPpass_pointer_scalar(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>{{.*}}) {
subroutine pass_pointer_scalar(i)
  integer, pointer :: i
  call takes_opt_scalar(i)
! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:         %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ptr<i32>) -> !fir.ref<i32>
! CHECK:         fir.call @_QPtakes_opt_scalar(%[[VAL_3]]) {{.*}}: (!fir.ref<i32>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_allocatable_scalar(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>{{.*}}) {
subroutine pass_allocatable_scalar(i)
  integer, allocatable :: i
  call takes_opt_scalar(i)
! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:         %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.heap<i32>) -> !fir.ref<i32>
! CHECK:         fir.call @_QPtakes_opt_scalar(%[[VAL_3]]) {{.*}}: (!fir.ref<i32>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_pointer_scalar_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>{{.*}}) {
subroutine pass_pointer_scalar_char(c)
  character(:), pointer :: c
  call takes_opt_scalar_char(c)
! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:         %[[VAL_2:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
! CHECK:         %[[VAL_3:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ptr<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:         %[[VAL_5:.*]] = fir.emboxchar %[[VAL_4]], %[[VAL_2]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:         fir.call @_QPtakes_opt_scalar_char(%[[VAL_5]]) {{.*}}: (!fir.boxchar<1>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_allocatable_scalar_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>{{.*}}) {
subroutine pass_allocatable_scalar_char(c)
  character(:), allocatable :: c
  call takes_opt_scalar_char(c)
! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:         %[[VAL_2:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
! CHECK:         %[[VAL_3:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:         %[[VAL_5:.*]] = fir.emboxchar %[[VAL_4]], %[[VAL_2]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:         fir.call @_QPtakes_opt_scalar_char(%[[VAL_5]]) {{.*}}: (!fir.boxchar<1>) -> ()
end subroutine

! -----------------------------------------------------------------------------
!     Test passing non contiguous pointers to explicit shape optional
! -----------------------------------------------------------------------------
! The pointer descriptor can be unconditionally read, but the copy-in/copy-out
! must be conditional on the pointer association status in order to get the
! correct present/absent aspect.

! CHECK-LABEL: func @_QMoptional_testsPpass_pointer_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}}) {
subroutine pass_pointer_array(i)
  real, pointer :: i(:)
  call takes_opt_explicit_shape(i)
! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
! CHECK:         %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:         %[[box:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:         %[[box_none:.*]] = fir.convert %[[box]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<none>
! CHECK:         %[[is_contiguous:.*]] = fir.call @_FortranAIsContiguous(%[[box_none]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK:         %[[VAL_9:.*]] = fir.if %[[VAL_5]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:           %[[addr:.*]] = fir.if %[[is_contiguous]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:             %[[box_addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:             fir.result %[[box_addr]] : !fir.heap<!fir.array<?xf32>>
! CHECK:           } else {
! CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_11:.*]]:3 = fir.box_dims %[[box]], %[[VAL_10]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_12:.*]] = fir.allocmem !fir.array<?xf32>, %[[VAL_11]]#1 {uniq_name = ".copyinout"}
! CHECK:           %[[VAL_20:.*]] = fir.do_loop {{.*}} {
! CHECK:           }
! CHECK:           fir.array_merge_store %{{.*}}, %[[VAL_20]] to %[[VAL_12]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>
! CHECK:           fir.result %[[VAL_12]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         } else {
! CHECK:           %[[VAL_26:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK:           fir.result %[[VAL_26]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         }
! CHECK:         %[[not_contiguous:.*]] = arith.cmpi eq, %[[is_contiguous]], %false : i1
! CHECK:         %[[and:.*]] = arith.andi %[[VAL_5]], %[[not_contiguous]] : i1
! CHECK:         %[[VAL_29:.*]] = fir.convert %[[VAL_9]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:         fir.call @_QPtakes_opt_explicit_shape(%[[VAL_29]]) {{.*}}: (!fir.ref<!fir.array<100xf32>>) -> ()
! CHECK:         fir.if %[[and]] {
! CHECK:           %[[VAL_40:.*]] = fir.do_loop {{.*}} {
! CHECK:           }
! CHECK:           fir.array_merge_store %{{.*}}, %[[VAL_40]] to %[[box]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:           fir.freemem %[[VAL_9]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         }
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_pointer_array_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>{{.*}}) {
subroutine pass_pointer_array_char(c)
  character(:), pointer :: c(:)
  call takes_opt_explicit_shape_char(c)
! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:         %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> !fir.ptr<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ptr<!fir.array<?x!fir.char<1,?>>>) -> i64
! CHECK:         %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:         %[[box_none:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> !fir.box<none>
! CHECK:         %[[is_contiguous:.*]] = fir.call @_FortranAIsContiguous(%[[box_none]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK:         %[[VAL_9:.*]] = fir.if %[[VAL_5]] -> (!fir.heap<!fir.array<?x!fir.char<1,?>>>) {
! CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_10]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_12:.*]] = fir.box_elesize %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> index
! CHECK:           %[[VAL_13:.*]] = fir.allocmem !fir.array<?x!fir.char<1,?>>(%[[VAL_12]] : index), %[[VAL_11]]#1 {uniq_name = ".copyinout"}
! CHECK:           %[[VAL_21:.*]] = fir.do_loop {{.*}} {
! CHECK:           }
! CHECK:           fir.array_merge_store %{{.*}}, %[[VAL_21]] to %[[VAL_13]] typeparams %[[VAL_12]] : !fir.array<?x!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>, !fir.heap<!fir.array<?x!fir.char<1,?>>>, index
! CHECK:           fir.result %[[VAL_13]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:         } else {
! CHECK:           %[[VAL_46:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:           fir.result %[[VAL_46]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:         }
! CHECK:         %[[VAL_47:.*]] = fir.box_elesize %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> index
! CHECK:         %[[not_contiguous:.*]] = arith.cmpi eq, %[[is_contiguous]], %false : i1
! CHECK:         %[[and:.*]] = arith.andi %[[VAL_5]], %[[not_contiguous]] : i1
! CHECK:         %[[VAL_50:.*]] = fir.convert %[[VAL_9]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:         %[[VAL_52:.*]] = fir.emboxchar %[[VAL_50]], %[[VAL_47]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:         fir.call @_QPtakes_opt_explicit_shape_char(%[[VAL_52]]) {{.*}}: (!fir.boxchar<1>) -> ()
! CHECK:         fir.if %[[and]] {
! CHECK:           %[[VAL_62:.*]] = fir.do_loop {{.*}} {
! CHECK:           }
! CHECK:           fir.array_merge_store %{{.*}}, %[[VAL_62]] to %[[VAL_6]] : !fir.array<?x!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>, !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
! CHECK:           fir.freemem %[[VAL_9]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:         }
! CHECK:         return
! CHECK:       }
end subroutine

! This case is bit special because the pointer is not a symbol but a function
! result. Test that the copy-in/copy-out is the same as with normal pointers.

! CHECK-LABEL: func @_QMoptional_testsPforward_pointer_array() {
subroutine forward_pointer_array()
  call takes_opt_explicit_shape(returns_pointer())
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {bindc_name = ".result"}
! CHECK:         %[[VAL_1:.*]] = fir.call @_QPreturns_pointer() {{.*}}: () -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:         fir.save_result %[[VAL_1]] to %[[VAL_0]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
! CHECK:         %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:         %[[is_contiguous:.*]] = fir.call @_FortranAIsContiguous(%{{.*}}) {{.*}}: (!fir.box<none>) -> i1
! CHECK:         %[[VAL_7:.*]] = fir.if %[[VAL_6]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:           %[[VAL_10:.*]] = fir.allocmem !fir.array<?xf32>
! CHECK:           fir.do_loop {{.*}} {
! CHECK:           }
! CHECK:           fir.result %[[VAL_10]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         } else {
! CHECK:           %[[VAL_11:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK:           fir.result %[[VAL_11]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         }
! CHECK:         %[[not_contiguous:.*]] = arith.cmpi eq, %[[is_contiguous]], %false : i1
! CHECK:         %[[and:.*]] = arith.andi %[[VAL_6]], %[[not_contiguous]] : i1
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_7]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:         fir.call @_QPtakes_opt_explicit_shape(%[[VAL_14]]) {{.*}}: (!fir.ref<!fir.array<100xf32>>) -> ()
! CHECK:         fir.if %[[and]] {
! CHECK:           fir.do_loop {{.*}} {
! CHECK:           }
! CHECK:           fir.freemem %[[VAL_7]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         }
end subroutine

! -----------------------------------------------------------------------------
!    Test passing assumed shape optional to explicit shape optional
! -----------------------------------------------------------------------------
! The fix.box can only be read if the assumed shape is present,
! and the copy-in/copy-out must also be conditional on the assumed
! shape presence.

! CHECK-LABEL: func @_QMoptional_testsPpass_opt_assumed_shape(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
subroutine pass_opt_assumed_shape(x)
  real, optional :: x(:)
  call takes_opt_explicit_shape(x)
! CHECK:         %[[VAL_1:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:         %[[VAL_2:.*]] = fir.zero_bits !fir.ref<!fir.array<?xf32>>
! CHECK:         %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_5:.*]] = fir.embox %[[VAL_2]](%[[VAL_4]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:         %[[VAL_6:.*]] = arith.select %[[VAL_1]], %[[VAL_0]], %[[VAL_5]] : !fir.box<!fir.array<?xf32>>
! CHECK:         %[[is_contiguous:.*]] = fir.call @_FortranAIsContiguous(%{{.*}}) {{.*}}: (!fir.box<none>) -> i1
! CHECK:         %[[VAL_7:.*]] = fir.if %[[VAL_1]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_8]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_10:.*]] = fir.allocmem !fir.array<?xf32>, %[[VAL_9]]#1 {uniq_name = ".copyinout"}
! CHECK:           %[[VAL_17:.*]] = fir.do_loop {{.*}} {
! CHECK:           }
! CHECK:           fir.array_merge_store %{{.*}}, %[[VAL_17]] to %[[VAL_10]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>
! CHECK:           fir.result %[[VAL_10]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         } else {
! CHECK:           %[[VAL_23:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK:           fir.result %[[VAL_23]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         }
! CHECK:         %[[not_contiguous:.*]] = arith.cmpi eq, %[[is_contiguous]], %false : i1
! CHECK:         %[[and:.*]] = arith.andi %[[VAL_1]], %[[not_contiguous]] : i1
! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_27:.*]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:         fir.call @_QPtakes_opt_explicit_shape(%[[VAL_26]]) {{.*}}: (!fir.ref<!fir.array<100xf32>>) -> ()
! CHECK:         fir.if %[[and]] {
! CHECK:           %[[VAL_36:.*]] = fir.do_loop {{.*}} { 
! CHECK:           }
! CHECK:           fir.array_merge_store %{{.*}}, %[[VAL_36]] to %[[VAL_6]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.box<!fir.array<?xf32>>
! CHECK:           fir.freemem %[[VAL_27]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         }
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_opt_assumed_shape_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c", fir.optional}) {
subroutine pass_opt_assumed_shape_char(c)
  character(*), optional :: c(:)
  call takes_opt_explicit_shape_char(c)
! CHECK:         %[[VAL_1:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> i1
! CHECK:         %[[VAL_2:.*]] = fir.zero_bits !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_6:.*]] = fir.embox %[[VAL_2]](%[[VAL_4]]) typeparams %[[VAL_5]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_7:.*]] = arith.select %[[VAL_1]], %[[VAL_0]], %[[VAL_6]] : !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[box_none:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<none>
! CHECK:         %[[is_contiguous:.*]] = fir.call @_FortranAIsContiguous(%[[box_none]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK:         %[[VAL_8:.*]] = fir.if %[[VAL_1]] -> (!fir.heap<!fir.array<?x!fir.char<1,?>>>) {
! CHECK:         %[[addr:.*]] = fir.if %[[is_contiguous]] -> (!fir.heap<!fir.array<?x!fir.char<1,?>>>) {
! CHECK:           %[[res:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:           fir.result %[[res]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:         } else {
! CHECK:           %[[box_elesize:.*]] = fir.box_elesize %[[VAL_7]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:           %[[temp:.*]] = fir.allocmem !fir.array<?x!fir.char<1,?>>(%[[box_elesize]] : index), %{{.*}}#1 {uniq_name = ".copyinout"}
! CHECK:           %[[VAL_19:.*]] = fir.do_loop  {{.*}} {
! CHECK:           }
! CHECK:           fir.array_merge_store %{{.*}}, %[[VAL_19]] to %[[temp]] typeparams %[[box_elesize]] : !fir.array<?x!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>, !fir.heap<!fir.array<?x!fir.char<1,?>>>, index
! CHECK:           fir.result %[[VAL_12]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:         } else {
! CHECK:           %[[VAL_44:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:           fir.result %[[VAL_44]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:         }
! CHECK:         %[[VAL_45:.*]] = fir.box_elesize %[[VAL_7]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:         %[[not_contiguous:.*]] = arith.cmpi eq, %[[is_contiguous]], %false : i1
! CHECK:         %[[and:.*]] = arith.andi %[[VAL_1]], %[[not_contiguous]] : i1
! CHECK:         %[[VAL_48:.*]] = fir.convert %[[VAL_49:.*]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:         %[[VAL_50:.*]] = fir.emboxchar %[[VAL_48]], %[[VAL_45]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:         fir.call @_QPtakes_opt_explicit_shape_char(%[[VAL_50]]) {{.*}}: (!fir.boxchar<1>) -> ()
! CHECK:         fir.if %[[and]] {
! CHECK:           %[[VAL_59:.*]] = fir.do_loop {{.*}} {
! CHECK:           fir.array_merge_store %{{.*}}, %[[VAL_59]] to %[[VAL_7]] : !fir.array<?x!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:           fir.freemem %[[VAL_49]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:         }
end subroutine

! -----------------------------------------------------------------------------
!    Test passing contiguous optional assumed shape to explicit shape optional
! -----------------------------------------------------------------------------
! The fix.box can only be read if the assumed shape is present.
! There should be no copy-in/copy-out

! CHECK-LABEL: func @_QMoptional_testsPpass_opt_contiguous_assumed_shape(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous, fir.optional}) {
subroutine pass_opt_contiguous_assumed_shape(x)
  real, optional, contiguous :: x(:)
  call takes_opt_explicit_shape(x)
! CHECK:         %[[VAL_1:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:         %[[VAL_2:.*]] = fir.zero_bits !fir.ref<!fir.array<?xf32>>
! CHECK:         %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_5:.*]] = fir.embox %[[VAL_2]](%[[VAL_4]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:         %[[VAL_6:.*]] = arith.select %[[VAL_1]], %[[VAL_0]], %[[VAL_5]] : !fir.box<!fir.array<?xf32>>
! CHECK:         %[[VAL_7:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:         fir.call @_QPtakes_opt_explicit_shape(%[[VAL_8]]) {{.*}}: (!fir.ref<!fir.array<100xf32>>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_opt_contiguous_assumed_shape_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c", fir.contiguous, fir.optional}) {
subroutine pass_opt_contiguous_assumed_shape_char(c)
  character(*), optional, contiguous :: c(:)
  call takes_opt_explicit_shape_char(c)
! CHECK:         %[[VAL_1:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> i1
! CHECK:         %[[VAL_2:.*]] = fir.zero_bits !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_6:.*]] = fir.embox %[[VAL_2]](%[[VAL_4]]) typeparams %[[VAL_5]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_7:.*]] = arith.select %[[VAL_1]], %[[VAL_0]], %[[VAL_6]] : !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_9:.*]] = fir.box_elesize %[[VAL_7]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:         %[[VAL_11:.*]] = fir.emboxchar %[[VAL_10]], %[[VAL_9]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:         fir.call @_QPtakes_opt_explicit_shape_char(%[[VAL_11]]) {{.*}}: (!fir.boxchar<1>) -> ()
end subroutine

! -----------------------------------------------------------------------------
!    Test passing allocatables and contiguous pointers to explicit shape optional
! -----------------------------------------------------------------------------
! The fix.box can be read and its address directly passed. There should be no
! copy-in/copy-out.

! CHECK-LABEL: func @_QMoptional_testsPpass_allocatable_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>{{.*}}) {
subroutine pass_allocatable_array(i)
  real, allocatable :: i(:)
  call takes_opt_explicit_shape(i)
! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:         fir.call @_QPtakes_opt_explicit_shape(%[[VAL_3]]) {{.*}}: (!fir.ref<!fir.array<100xf32>>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_allocatable_array_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>{{.*}}) {
subroutine pass_allocatable_array_char(c)
  character(:), allocatable :: c(:)
  call takes_opt_explicit_shape_char(c)
! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:         %[[VAL_2:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> index
! CHECK:         %[[VAL_3:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:         %[[VAL_5:.*]] = fir.emboxchar %[[VAL_4]], %[[VAL_2]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:         fir.call @_QPtakes_opt_explicit_shape_char(%[[VAL_5]]) {{.*}}: (!fir.boxchar<1>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_contiguous_pointer_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "i", fir.contiguous}) {
subroutine pass_contiguous_pointer_array(i)
  real, pointer, contiguous :: i(:)
  call takes_opt_explicit_shape(i)
! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ptr<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:         fir.call @_QPtakes_opt_explicit_shape(%[[VAL_3]]) {{.*}}: (!fir.ref<!fir.array<100xf32>>) -> ()
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_contiguous_pointer_array_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>> {fir.bindc_name = "c", fir.contiguous}) {
subroutine pass_contiguous_pointer_array_char(c)
  character(:), pointer, contiguous :: c(:)
  call takes_opt_explicit_shape_char(c)
! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:         %[[VAL_2:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> index
! CHECK:         %[[VAL_3:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> !fir.ptr<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ptr<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:         %[[VAL_5:.*]] = fir.emboxchar %[[VAL_4]], %[[VAL_2]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:         fir.call @_QPtakes_opt_explicit_shape_char(%[[VAL_5]]) {{.*}}: (!fir.boxchar<1>) -> ()
end subroutine

! -----------------------------------------------------------------------------
!    Test passing assumed shape optional to explicit shape optional with intents
! -----------------------------------------------------------------------------
! The fix.box can only be read if the assumed shape is present,
! and the copy-in/copy-out must also be conditional on the assumed
! shape presence. For intent(in), there should be no copy-out while for
! intent(out), there should be no copy-in.

! CHECK-LABEL: func @_QMoptional_testsPpass_opt_assumed_shape_to_intentin(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
subroutine pass_opt_assumed_shape_to_intentin(x)
  real, optional :: x(:)
  call takes_opt_explicit_shape_intentin(x)
! CHECK:         %[[VAL_1:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:         %[[VAL_2:.*]] = fir.zero_bits !fir.ref<!fir.array<?xf32>>
! CHECK:         %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_5:.*]] = fir.embox %[[VAL_2]](%[[VAL_4]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:         %[[VAL_6:.*]] = arith.select %[[VAL_1]], %[[VAL_0]], %[[VAL_5]] : !fir.box<!fir.array<?xf32>>
! CHECK:         %[[box_none:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK:         %[[is_contiguous:.*]] = fir.call @_FortranAIsContiguous(%[[box_none]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK:         %[[VAL_7:.*]] = fir.if %[[VAL_1]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:           %[[VAL_10:.*]] = fir.allocmem !fir.array<?xf32>
! CHECK:           fir.do_loop {{.*}} {
! CHECK:           }
! CHECK:           fir.result %[[VAL_10]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         } else {
! CHECK:           %[[VAL_23:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK:           fir.result %[[VAL_23]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         }
! CHECK:         %[[not_contiguous:.*]] = arith.cmpi eq, %[[is_contiguous]], %false : i1
! CHECK:         %[[and:.*]] = arith.andi %[[VAL_1]], %[[not_contiguous]] : i1
! CHECK:         %[[VAL_24:.*]] = fir.convert %[[VAL_7]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:         fir.call @_QPtakes_opt_explicit_shape_intentin(%[[VAL_24]]) {{.*}}: (!fir.ref<!fir.array<100xf32>>) -> ()
! CHECK:         fir.if %[[and]] {
! CHECK-NOT:       fir.do_loop
! CHECK:           fir.freemem %[[VAL_7]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         }
end subroutine

! CHECK-LABEL: func @_QMoptional_testsPpass_opt_assumed_shape_to_intentout(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
subroutine pass_opt_assumed_shape_to_intentout(x)
  real, optional :: x(:)
  call takes_opt_explicit_shape_intentout(x)
! CHECK:         %[[VAL_1:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:         %[[VAL_2:.*]] = fir.zero_bits !fir.ref<!fir.array<?xf32>>
! CHECK:         %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_5:.*]] = fir.embox %[[VAL_2]](%[[VAL_4]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:         %[[VAL_6:.*]] = arith.select %[[VAL_1]], %[[VAL_0]], %[[VAL_5]] : !fir.box<!fir.array<?xf32>>
! CHECK:         %[[box_none:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK:         %[[is_contiguous:.*]] = fir.call @_FortranAIsContiguous(%[[box_none]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK:         %[[VAL_7:.*]] = fir.if %[[VAL_1]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:           %[[VAL_10:.*]] = fir.allocmem !fir.array<?xf32>
! CHECK-NOT:       fir.do_loop
! CHECK:           fir.result %[[VAL_10]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         } else {
! CHECK:           %[[VAL_11:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK:           fir.result %[[VAL_11]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         }
! CHECK:         %[[not_contiguous:.*]] = arith.cmpi eq, %[[is_contiguous]], %false : i1
! CHECK:         %[[and:.*]] = arith.andi %[[VAL_1]], %[[not_contiguous]] : i1
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_7]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:         fir.call @_QPtakes_opt_explicit_shape_intentout(%[[VAL_14]]) {{.*}}: (!fir.ref<!fir.array<100xf32>>) -> ()
! CHECK:         fir.if %[[and]] {
! CHECK:           fir.do_loop {{.*}} {
! CHECK:           }
! CHECK:           fir.freemem %[[VAL_7]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         }
end subroutine
end module
