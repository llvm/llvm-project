! Test lowering of character function dummy procedure. The length must be
! passed along the function address.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! -----------------------------------------------------------------------------
!     Test passing a character function as dummy procedure
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPcst_len() {
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
! CHECK:  fir.call @_QPfoo1(%[[VAL_5]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
end subroutine

! CHECK-LABEL: func.func @_QPcst_len_array() {
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
! CHECK:  fir.call @_QPfoo1b(%[[VAL_5]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
  call foo1b(bar1_array)
end subroutine

! CHECK-LABEL: func.func @_QPcst_len_2() {
subroutine cst_len_2()
  character(7) :: bar2
  external :: bar2
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPbar2) : (!fir.ref<!fir.char<1,7>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]] = arith.constant 7 : i64
! CHECK:  %[[VAL_2:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<!fir.char<1,7>>, index) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_3:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_4:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_1]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPfoo2(%[[VAL_5]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
  call foo2(bar2)
end subroutine

! CHECK-LABEL: func.func @_QPdyn_len(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32>{{.*}}) {
subroutine dyn_len(n)
  integer :: n
  character(n) :: bar3
  external :: bar3
! CHECK:  %[[VAL_ARG:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}
! CHECK:  %[[VAL_1:.*]] = fir.address_of(@_QPbar3) : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_ARG]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i32) -> i64
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_5:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:  %[[VAL_6:.*]] = arith.select %[[VAL_5]], %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:  %[[VAL_7:.*]] = fir.emboxproc %[[VAL_1]] : ((!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_8:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_9:.*]] = fir.insert_value %[[VAL_8]], %[[VAL_7]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_10:.*]] = fir.insert_value %[[VAL_9]], %[[VAL_6]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPfoo3(%[[VAL_10]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
  call foo3(bar3)
end subroutine

! CHECK-LABEL: func.func @_QPcannot_compute_len_yet() {
subroutine cannot_compute_len_yet()
  interface
    function bar4(n)
      integer :: n
      character(n) :: bar4
    end function
  end interface
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPbar4) : (!fir.ref<!fir.char<1,?>>, index, !fir.ref<i32>) -> !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_2:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<!fir.char<1,?>>, index, !fir.ref<i32>) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_1]] : (index) -> i64
! CHECK:  %[[VAL_4:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_6:.*]] = fir.insert_value %[[VAL_5]], %[[VAL_3]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPfoo4(%[[VAL_6]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
  call foo4(bar4)
end subroutine

! CHECK-LABEL: func.func @_QPcannot_compute_len_yet_2() {
subroutine cannot_compute_len_yet_2()
  character(*) :: bar5
  external :: bar5
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPbar5) : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_2:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_1]] : (index) -> i64
! CHECK:  %[[VAL_4:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_6:.*]] = fir.insert_value %[[VAL_5]], %[[VAL_3]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPfoo5(%[[VAL_6]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
  call foo5(bar5)
end subroutine

! CHECK-LABEL: func.func @_QPforward_incoming_length(
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
! CHECK:  fir.call @_QPfoo6(%[[VAL_5]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
  call foo6(bar6)
end subroutine

! CHECK-LABEL: func.func @_QPoverride_incoming_length(
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
! CHECK:  fir.call @_QPfoo7(%[[VAL_5]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
  call foo7(bar7)
end subroutine

! -----------------------------------------------------------------------------
!     Test calling character dummy function
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPcall_assumed_length(
! CHECK-SAME: %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}) {
subroutine call_assumed_length(bar8)
  character(*) :: bar8
  external :: bar8
! CHECK:  %[[VAL_CONST:.*]] = arith.constant 42 : i32
! CHECK:  %[[VAL_ASSOC:.*]]:3 = hlfir.associate %[[VAL_CONST]] {{.*}}
! CHECK:  %[[VAL_3:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[WAL_2:.*]] = fir.box_addr %[[VAL_3]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_4:.*]] = fir.extract_value %[[VAL_0]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> i64
! CHECK:  %[[EMBOX:.*]] = fir.emboxproc %[[WAL_2]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:  %[[TUPLE:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[INS1:.*]] = fir.insert_value %[[TUPLE]], %[[EMBOX]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[INS2:.*]] = fir.insert_value %[[INS1]], %[[VAL_4]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_EXT0:.*]] = fir.extract_value %[[INS2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_EXT1:.*]] = fir.extract_value %[[INS2]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> i64
! CHECK:  %[[VAL_6:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_EXT1]] : i64) {bindc_name = ".result"}
! CHECK:  %[[VAL_7:.*]] = fir.box_addr %[[VAL_EXT0]] : (!fir.boxproc<() -> ()>) -> ((!fir.ref<!fir.char<1,?>>, index, !fir.ref<i32>) -> !fir.boxchar<1>)
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_EXT1]] : (i64) -> index
! CHECK:  fir.call %[[VAL_7]](%[[VAL_6]], %[[VAL_8]], %[[VAL_ASSOC]]#0) {{.*}}: (!fir.ref<!fir.char<1,?>>, index, !fir.ref<i32>) -> !fir.boxchar<1>
  call test(bar8(42))
end subroutine

! CHECK-LABEL: func.func @_QPcall_explicit_length(
! CHECK-SAME: %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}) {
subroutine call_explicit_length(bar9)
  character(7) :: bar9
  external :: bar9
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.char<1,7> {bindc_name = ".result"}
! CHECK:  %[[VAL_CONST:.*]] = arith.constant 42 : i32
! CHECK:  %[[VAL_ASSOC:.*]]:3 = hlfir.associate %[[VAL_CONST]] {{.*}}
! CHECK:  %[[VAL_4:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[WAL_1:.*]] = fir.box_addr %[[VAL_4]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_5:.*]] = arith.constant 7 : i64
! CHECK:  %[[EMBOX:.*]] = fir.emboxproc %[[WAL_1]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:  %[[TUPLE:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[INS1:.*]] = fir.insert_value %[[TUPLE]], %[[EMBOX]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[INS2:.*]] = fir.insert_value %[[INS1]], %[[VAL_5]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_EXT0:.*]] = fir.extract_value %[[INS2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_5_NEW:.*]] = arith.constant 7 : i64
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5_NEW]] : (i64) -> index
! CHECK:  %[[C0:.*]] = arith.constant 0 : index
! CHECK:  %[[CMPI:.*]]  = arith.cmpi sgt, %[[VAL_6]], %[[C0]] : index
! CHECK:  %[[SELECT:.*]] = arith.select %[[CMPI]], %[[VAL_6]], %[[C0]] : index
! CHECK:  %[[VAL_7:.*]] = fir.box_addr %[[VAL_EXT0]] : (!fir.boxproc<() -> ()>) -> ((!fir.ref<!fir.char<1,7>>, index, !fir.ref<i32>) -> !fir.boxchar<1>)
! CHECK:  fir.call %[[VAL_7]](%[[VAL_1]], %[[SELECT]], %[[VAL_ASSOC]]#0) {{.*}}: (!fir.ref<!fir.char<1,7>>, index, !fir.ref<i32>) -> !fir.boxchar<1>
  call test(bar9(42))
end subroutine

! CHECK-LABEL: func.func @_QPcall_explicit_length_with_iface(
! CHECK-SAME: %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}) {
subroutine call_explicit_length_with_iface(bar10)
  interface
    function bar10(n)
      integer(8) :: n
      character(n) :: bar10
    end function
  end interface
! CHECK:  %[[VAL_2:.*]] = arith.constant 42 : i64
! CHECK:  %[[VAL_ASSOC:.*]]:3 = hlfir.associate %[[VAL_2]] {{.*}}
! CHECK:  %[[VAL_DECL:.*]]:2 = hlfir.declare %[[VAL_ASSOC]]#0 {{.*}}
! CHECK:  %[[VAL_3:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[WAL_1:.*]] = fir.box_addr %[[VAL_3]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_DECL]]#0 : !fir.ref<i64>
! CHECK:  %[[C0:.*]] = arith.constant 0 : i64
! CHECK:  %[[COMPI:.*]] = arith.cmpi sgt, %[[VAL_4]], %[[C0]] : i64
! CHECK:  %[[SELECT:.*]] = arith.select %[[COMPI]], %[[VAL_4]], %[[C0]] : i64
! CHECK:  %[[EMBOX:.*]] = fir.emboxproc %[[WAL_1]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:  %[[TUPLE:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[INS1:.*]] = fir.insert_value %[[TUPLE]], %[[EMBOX]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[INS2:.*]] = fir.insert_value %[[INS1]], %[[SELECT]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_EXT0:.*]] = fir.extract_value %[[INS2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_13:.*]] = fir.load %[[VAL_DECL]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:  %[[C0_IDX:.*]] = arith.constant 0 : index
! CHECK:  %[[CMP_IDX:.*]] = arith.cmpi sgt, %[[VAL_14]], %[[C0_IDX]] : index
! CHECK:  %[[SELECT_IDX:.*]] = arith.select %[[CMP_IDX]], %[[VAL_14]], %[[C0_IDX]] : index
! CHECK:  %[[VAL_7:.*]] = fir.alloca !fir.char<1,?>(%[[SELECT_IDX]] : index) {bindc_name = ".result"}
! CHECK:  %[[VAL_8:.*]] = fir.box_addr %[[VAL_EXT0]] : (!fir.boxproc<() -> ()>) -> ((!fir.ref<!fir.char<1,?>>, index, !fir.ref<i64>) -> !fir.boxchar<1>)
! CHECK:  fir.call %[[VAL_8]](%[[VAL_7]], %[[SELECT_IDX]], %[[VAL_ASSOC]]#0) {{.*}}: (!fir.ref<!fir.char<1,?>>, index, !fir.ref<i64>) -> !fir.boxchar<1>
  call test(bar10(42_8))
end subroutine


! CHECK-LABEL: func.func @_QPhost(
! CHECK-SAME:  %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64>
subroutine host(f)
  character*(*) :: f
  external :: f
  ! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1:.*]], %{{.*}} : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>>>, i32) -> !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
  ! CHECK:  fir.store %[[VAL_0]] to %[[VAL_3]] : !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
  ! CHECK: fir.call @_QFhostPintern(%[[VAL_1]])
  call intern()
contains
! CHECK-LABEL: func.func private @_QFhostPintern(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>>> {fir.host_assoc})
  subroutine intern()
! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>>>, i32) -> !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
! CHECK:  %[[VAL_4:.*]] = fir.extract_value %[[VAL_3]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[WAL_1:.*]] = fir.box_addr %[[VAL_4]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_5:.*]] = fir.extract_value %[[VAL_3]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> i64
! CHECK:  %[[VAL_6:.*]] = fir.emboxproc %[[WAL_1]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_7:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_6]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_9:.*]] = fir.insert_value %[[VAL_8]], %[[VAL_5]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_10:.*]] = fir.extract_value %[[VAL_9]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_11:.*]] = fir.extract_value %[[VAL_9]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> i64
! CHECK:  %[[VAL_12:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK:  %[[VAL_13:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_11]] : i64) {bindc_name = ".result"}
! CHECK:  %[[VAL_14:.*]] = fir.box_addr %[[VAL_10]] : (!fir.boxproc<() -> ()>) -> ((!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>)
! CHECK:  %[[VAL_15:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:  fir.call %[[VAL_14]](%[[VAL_13]], %[[VAL_15]]) {{.*}}: (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
    call test(f())
  end subroutine
end subroutine

! CHECK-LABEL: func.func @_QPhost2(
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
! CHECK-LABEL: func.func private @_QFhost2Pintern(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>>> {fir.host_assoc})
  subroutine intern()
    ! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.char<1,42> {bindc_name = ".result"}
    ! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : i32
    ! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>>>, i32) -> !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
    ! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
    ! CHECK:  %[[VAL_5:.*]] = fir.extract_value %[[VAL_4]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
    ! CHECK:  %[[WAL_1:.*]] = fir.box_addr %[[VAL_5]] : (!fir.boxproc<() -> ()>) -> (() -> ())
    ! CHECK:  %[[VAL_6:.*]] = arith.constant 42 : i64
    ! CHECK:  %[[VAL_NEW_EMBOX:.*]] = fir.emboxproc %[[WAL_1]] : (() -> ()) -> !fir.boxproc<() -> ()>
    ! CHECK:  %[[VAL_NEW_TUPLE:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
    ! CHECK:  %[[VAL_INS1:.*]] = fir.insert_value %[[VAL_NEW_TUPLE]], %[[VAL_NEW_EMBOX]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
    ! CHECK:  %[[VAL_INS2:.*]] = fir.insert_value %[[VAL_INS1]], %[[VAL_6]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
    ! CHECK:  %[[VAL_EXT1:.*]] = fir.extract_value %[[VAL_INS2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
    ! CHECK:  %[[VAL_CONST_42:.*]] = arith.constant 42 : i64
    ! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_CONST_42]] : (i64) -> index
    ! CHECK:  %[[C0:.*]] = arith.constant 0 : index
    ! CHECK:  %[[CMPI:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[C0]] : index
    ! CHECK:  %[[SELECT:.*]] = arith.select %[[CMPI]], %[[VAL_7]], %[[C0]] : index
    ! CHECK:  %[[VAL_9:.*]] = fir.box_addr %[[VAL_EXT1]] : (!fir.boxproc<() -> ()>) -> ((!fir.ref<!fir.char<1,42>>, index) -> !fir.boxchar<1>)
    ! CHECK:  fir.call %[[VAL_9]](%[[VAL_1]], %[[SELECT]]) {{.*}}: (!fir.ref<!fir.char<1,42>>, index) -> !fir.boxchar<1>
    call test(f())
  end subroutine
end subroutine
