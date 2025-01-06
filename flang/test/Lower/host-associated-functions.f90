! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! Test calling functions whose result interface is evaluated on the call site
! and where the calls are located in an internal procedure while the
! interface is defined in the host procedure.

! CHECK-LABEL: func @_QPcapture_char_func_dummy(
! CHECK-SAME:  %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
subroutine capture_char_func_dummy(char_func_dummy, n)
  character(n),external :: char_func_dummy
  ! CHECK:  %[[VAL_2:.*]] = fir.alloca tuple<tuple<!fir.boxproc<() -> ()>, i64>, !fir.ref<i32>>
  ! CHECK:  %[[VAL_3:.*]] = arith.constant 0 : i32
  ! CHECK:  %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_3]] : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>, !fir.ref<i32>>>, i32) -> !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
  ! CHECK:  fir.store %[[VAL_0]] to %[[VAL_4]] : !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
  ! CHECK:  %[[VAL_5:.*]] = arith.constant 1 : i32
  ! CHECK:  %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_5]] : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>, !fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK:  fir.store %[[VAL_1]] to %[[VAL_6]] : !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK:  fir.call @_QFcapture_char_func_dummyPinternal(%[[VAL_2]]) {{.*}}: (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>, !fir.ref<i32>>>) -> ()
  call internal()
contains
  ! CHECK-LABEL: func private @_QFcapture_char_func_dummyPinternal(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>, !fir.ref<i32>>> {fir.host_assoc}) attributes {fir.host_symbol = {{.*}}, llvm.linkage = #llvm.linkage<internal>} {
  subroutine internal()
  ! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : i32
  ! CHECK:  %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>, !fir.ref<i32>>>, i32) -> !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
  ! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
  ! CHECK:  %[[VAL_4:.*]] = arith.constant 1 : i32
  ! CHECK:  %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_4]] : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>, !fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_5]] : !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK:  %[[VAL_12:.*]] = fir.extract_value %[[VAL_3]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
  ! CHECK:  %[[VAL_13:.*]] = fir.box_addr %[[VAL_12]] : (!fir.boxproc<() -> ()>) -> (() -> ())
  ! CHECK:  %[[VAL_14:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
  ! CHECK:  %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> i64
  ! CHECK:  %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i64) -> index
  ! CHECK:  %[[C0:.*]] = arith.constant 0 : index
  ! CHECK:  %[[CMPI:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[C0]] : index
  ! CHECK:  %[[SELECT:.*]] = arith.select %[[CMPI]], %[[VAL_16]], %[[C0]] : index
  ! CHECK:  %[[VAL_17:.*]] = llvm.intr.stacksave : !llvm.ptr
  ! CHECK:  %[[VAL_18:.*]] = fir.alloca !fir.char<1,?>(%[[SELECT]] : index) {bindc_name = ".result"}
  ! CHECK:  %[[VAL_19:.*]] = fir.convert %[[VAL_13]] : (() -> ()) -> ((!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>)
  ! CHECK:  %[[VAL_20:.*]] = fir.call %[[VAL_19]](%[[VAL_18]], %[[SELECT]]) {{.*}}: (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
   print *, char_func_dummy()
  end subroutine
end subroutine

! CHECK-LABEL: func @_QPcapture_char_func_assumed_dummy(
! CHECK-SAME:  %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}) {
subroutine capture_char_func_assumed_dummy(char_func_dummy)
  character(*),external :: char_func_dummy
! CHECK:  %[[VAL_1:.*]] = fir.alloca tuple<tuple<!fir.boxproc<() -> ()>, i64>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>>>, i32) -> !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
! CHECK:  fir.store %[[VAL_0]] to %[[VAL_3]] : !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
! CHECK:  fir.call @_QFcapture_char_func_assumed_dummyPinternal(%[[VAL_1]]) {{.*}}: (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>>>) -> ()
  call internal()
contains
! CHECK-LABEL: func private @_QFcapture_char_func_assumed_dummyPinternal(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>>> {fir.host_assoc}) attributes {fir.host_symbol = {{.*}}, llvm.linkage = #llvm.linkage<internal>} {
  subroutine internal()
! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>>>, i32) -> !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
! CHECK:  %[[VAL_9:.*]] = fir.extract_value %[[VAL_3]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_10:.*]] = fir.box_addr %[[VAL_9]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_11:.*]] = fir.extract_value %[[VAL_3]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> i64
! CHECK:  %[[VAL_12:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK:  %[[VAL_13:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_11]] : i64) {bindc_name = ".result"}
! CHECK:  %[[VAL_14:.*]] = fir.convert %[[VAL_10]] : (() -> ()) -> ((!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>)
! CHECK:  %[[VAL_15:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:  %[[VAL_16:.*]] = fir.call %[[VAL_14]](%[[VAL_13]], %[[VAL_15]]) {{.*}}: (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
   print *, char_func_dummy()
  end subroutine
end subroutine

! CHECK-LABEL: func @_QPcapture_char_func(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
subroutine capture_char_func(n)
  character(n), external :: char_func
! CHECK:  %[[VAL_1:.*]] = fir.alloca tuple<!fir.ref<i32>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  fir.store %[[VAL_0]] to %[[VAL_3]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  fir.call @_QFcapture_char_funcPinternal(%[[VAL_1]]) {{.*}}: (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
  call internal()
contains
! CHECK-LABEL: func private @_QFcapture_char_funcPinternal(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc})
  subroutine internal()
   print *, char_func()
  end subroutine
end subroutine

! CHECK-LABEL: func @_QPcapture_array_func(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
subroutine capture_array_func(n)
  integer :: n
  interface
  function array_func()
    import :: n
    integer :: array_func(n)
  end function
  end interface
! CHECK:  %[[VAL_1:.*]] = fir.alloca tuple<!fir.ref<i32>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  fir.store %[[VAL_0]] to %[[VAL_3]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  fir.call @_QFcapture_array_funcPinternal(%[[VAL_1]]) {{.*}}: (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
  call internal()
contains
  subroutine internal()
! CHECK-LABEL: func private @_QFcapture_array_funcPinternal(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) attributes {fir.host_symbol = {{.*}}, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  %[[VAL_9:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> i64
! CHECK:  %[[VAL_11:.*]] = arith.constant 1 : i64
! CHECK:  %[[VAL_12:.*]] = arith.subi %[[VAL_10]], %[[VAL_11]] : i64
! CHECK:  %[[VAL_13:.*]] = arith.constant 1 : i64
! CHECK:  %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : i64
! CHECK:  %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> index
! CHECK:  %[[C0:.*]] = arith.constant 0 : index
! CHECK:  %[[CMPI:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[C0]] : index
! CHECK:  %[[SELECT:.*]] = arith.select %[[CMPI]], %[[VAL_15]], %[[C0]] : index
! CHECK:  %[[VAL_16:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK:  %[[VAL_17:.*]] = fir.alloca !fir.array<?xi32>, %[[SELECT]] {bindc_name = ".result"}
   print *, array_func()
  end subroutine
end subroutine

module define_char_func
  contains
  function return_char(n)
    integer :: n
    character(n) :: return_char
    return_char = "a"
  end function
end module

! CHECK-LABEL: func @_QPuse_module() {
subroutine use_module()
  ! verify there is no capture triggers by the interface.
  use define_char_func
! CHECK:  fir.call @_QFuse_modulePinternal() {{.*}}: () -> ()
  call internal()
  contains
! CHECK-LABEL: func private @_QFuse_modulePinternal() {{.*}} {
  subroutine internal()
    print *, return_char(42)
  end subroutine
end subroutine
