! Test lowering of character function dummy procedure. The length must be
! passed along the function address.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! -----------------------------------------------------------------------------
!     Test passing a character function as dummy procedure
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPcst_len
subroutine cst_len()
  interface
    character(7) function bar1()
    end function
  end interface
  call foo1(bar1)
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPbar1) : (!fir.ref<!fir.char<1,7>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]] = arith.constant 7 : i64
! CHECK:  %[[VAL_2:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<!fir.char<1,7>>, index) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_3:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_4:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_1]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPfoo1(%[[VAL_5]]) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
end subroutine

! CHECK-LABEL: func @_QPcst_len_array
subroutine cst_len_array()
  interface
    function bar1_array()
      character(7) :: bar1_array(10)
    end function
  end interface
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPbar1_array) : () -> !fir.array<10x!fir.char<1,7>>
! CHECK:  %[[VAL_1:.*]] = arith.constant 7 : i64
! CHECK:  %[[VAL_2:.*]] = fir.emboxproc %[[VAL_0]] : (() -> !fir.array<10x!fir.char<1,7>>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_3:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_4:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_1]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPfoo1b(%[[VAL_5]]) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
  call foo1b(bar1_array)
end subroutine

! CHECK-LABEL: func @_QPcst_len_2
subroutine cst_len_2()
  character(7) :: bar2
  external :: bar2
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPbar2) : (!fir.ref<!fir.char<1,7>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]] = arith.constant 7 : i64
! CHECK:  %[[VAL_2:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<!fir.char<1,7>>, index) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_3:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_4:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_1]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPfoo2(%[[VAL_5]]) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
  call foo2(bar2)
end subroutine

! CHECK-LABEL: func @_QPdyn_len(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32>{{.*}}) {
subroutine dyn_len(n)
  integer :: n
  character(n) :: bar3
  external :: bar3
! CHECK:  %[[VAL_1:.*]] = fir.address_of(@_QPbar3) : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i32) -> i64
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_5:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:  %[[VAL_6:.*]] = arith.select %[[VAL_5]], %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:  %[[VAL_7:.*]] = fir.emboxproc %[[VAL_1]] : ((!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_8:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_9:.*]] = fir.insert_value %[[VAL_8]], %[[VAL_7]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_10:.*]] = fir.insert_value %[[VAL_9]], %[[VAL_6]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPfoo3(%[[VAL_10]]) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
  call foo3(bar3)
end subroutine

! CHECK-LABEL: func @_QPcannot_compute_len_yet
subroutine cannot_compute_len_yet()
  interface
    function bar4(n)
      integer :: n
      character(n) :: bar4
    end function
  end interface
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPbar4) : (!fir.ref<!fir.char<1,?>>, index, !fir.ref<i32>) -> !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]] = arith.constant -1 : index
! CHECK:  %[[VAL_2:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<!fir.char<1,?>>, index, !fir.ref<i32>) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_1]] : (index) -> i64
! CHECK:  %[[VAL_4:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_6:.*]] = fir.insert_value %[[VAL_5]], %[[VAL_3]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPfoo4(%[[VAL_6]]) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
  call foo4(bar4)
end subroutine

! CHECK-LABEL: func @_QPcannot_compute_len_yet_2
subroutine cannot_compute_len_yet_2()
  character(*) :: bar5
  external :: bar5
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPbar5) : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]] = arith.constant -1 : index
! CHECK:  %[[VAL_2:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_1]] : (index) -> i64
! CHECK:  %[[VAL_4:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_6:.*]] = fir.insert_value %[[VAL_5]], %[[VAL_3]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPfoo5(%[[VAL_6]]) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
  call foo5(bar5)
end subroutine

! CHECK-LABEL: func @_QPforward_incoming_length
! CHECK-SAME: %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}) {
subroutine forward_incoming_length(bar6)
  character(*) :: bar6
  external :: bar6
! CHECK:  %[[VAL_1:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[WAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_2:.*]] = fir.extract_value %[[VAL_0]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> i64
! CHECK:  %[[WAL_1:.*]] = fir.emboxproc %[[WAL_2]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_3:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_4:.*]] = fir.insert_value %[[VAL_3]], %[[WAL_1]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_2]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPfoo6(%[[VAL_5]]) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
  call foo6(bar6)
end subroutine

! CHECK-LABEL: func @_QPoverride_incoming_length
! CHECK-SAME: %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}) {
subroutine override_incoming_length(bar7)
  character(7) :: bar7
  external :: bar7
! CHECK:  %[[VAL_1:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[WAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_2:.*]] = arith.constant 7 : i64
! CHECK:  %[[WAL_1:.*]] = fir.emboxproc %[[WAL_2]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_3:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_4:.*]] = fir.insert_value %[[VAL_3]], %[[WAL_1]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_2]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPfoo7(%[[VAL_5]]) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
  call foo7(bar7)
end subroutine

! -----------------------------------------------------------------------------
!     Test calling character dummy function
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPcall_explicit_length
! CHECK-SAME: %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}) {
subroutine call_explicit_length(bar9)
  character(7) :: bar9
  external :: bar9
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.char<1,7> {bindc_name = ".result"}
! CHECK:  %[[VAL_4:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[WAL_1:.*]] = fir.box_addr %[[VAL_4]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_5:.*]] = arith.constant 7 : i64
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[WAL_1]] : (() -> ()) -> ((!fir.ref<!fir.char<1,7>>, index, !fir.ref<i32>) -> !fir.boxchar<1>)
! CHECK:  fir.call %[[VAL_8]](%[[VAL_1]], %[[VAL_6]], %{{.*}}) : (!fir.ref<!fir.char<1,7>>, index, !fir.ref<i32>) -> !fir.boxchar<1>
  call test(bar9(42))
end subroutine

! CHECK-LABEL: func @_QPcall_explicit_length_with_iface
! CHECK-SAME: %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}) {
subroutine call_explicit_length_with_iface(bar10)
  interface
    function bar10(n)
      integer(8) :: n
      character(n) :: bar10
    end function
  end interface
! CHECK:  %[[VAL_1:.*]] = fir.alloca i64
! CHECK:  %[[VAL_2:.*]] = arith.constant 42 : i64
! CHECK:  fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<i64>
! CHECK:  %[[VAL_3:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[WAL_1:.*]] = fir.box_addr %[[VAL_3]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:  %[[VAL_6:.*]] = fir.call @llvm.stacksave() : () -> !fir.ref<i8>
! CHECK:  %[[VAL_7:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_5]] : index) {bindc_name = ".result"}
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[WAL_1]] : (() -> ()) -> ((!fir.ref<!fir.char<1,?>>, index, !fir.ref<i64>) -> !fir.boxchar<1>)
! CHECK:  fir.call %[[VAL_8]](%[[VAL_7]], %[[VAL_5]], %[[VAL_1]]) : (!fir.ref<!fir.char<1,?>>, index, !fir.ref<i64>) -> !fir.boxchar<1>
  call test(bar10(42_8))
end subroutine

! CHECK-LABEL: func @_QPhost2(
! CHECK-SAME:  %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc})
subroutine host2(f)
  ! Test that dummy length is overridden by local length even when used
  ! in the internal procedure. 
  character*(42) :: f
  external :: f
  ! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1:.*]], %{{.*}} : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>>>, i32) -> !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
  ! CHECK:  fir.store %[[VAL_0]] to %[[VAL_3]] : !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
  ! CHECK: fir.call @_QFhost2Pintern(%[[VAL_1]])
  call intern()
contains
! CHECK-LABEL: func @_QFhost2Pintern(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>>> {fir.host_assoc})
  subroutine intern()
    ! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.char<1,42> {bindc_name = ".result"}
    ! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : i32
    ! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>>>, i32) -> !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
    ! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
    ! CHECK:  %[[VAL_5:.*]] = fir.extract_value %[[VAL_4]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
    ! CHECK:  %[[WAL_1:.*]] = fir.box_addr %[[VAL_5]] : (!fir.boxproc<() -> ()>) -> (() -> ())
    ! CHECK:  %[[VAL_6:.*]] = arith.constant 42 : i64
    ! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
    ! CHECK:  %[[VAL_9:.*]] = fir.convert %[[WAL_1]] : (() -> ()) -> ((!fir.ref<!fir.char<1,42>>, index) -> !fir.boxchar<1>)
    ! CHECK:  fir.call %[[VAL_9]](%[[VAL_1]], %[[VAL_7]]) : (!fir.ref<!fir.char<1,42>>, index) -> !fir.boxchar<1>
    call test(f())
  end subroutine
end subroutine
