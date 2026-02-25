! Test calling functions whose result interface is evaluated on the call site
! and where the calls are located in an internal procedure while the
! interface is defined in the host procedure.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPcapture_char_func_dummy(
! CHECK-SAME:  %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc},
! CHECK-SAME:  %[[VAL_1_arg:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
subroutine capture_char_func_dummy(char_func_dummy, n)
  character(n),external :: char_func_dummy
  ! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_1_arg]] {{.*}}uniq_name = "_QFcapture_char_func_dummyEn"
  ! CHECK:  %[[VAL_2:.*]] = fir.alloca tuple<tuple<!fir.boxproc<() -> ()>, i64>, !fir.ref<i32>>
  ! CHECK:  %[[VAL_3:.*]] = arith.constant 0 : i32
  ! CHECK:  %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_3]] : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>, !fir.ref<i32>>>, i32) -> !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
  ! CHECK:  fir.store %[[VAL_0]] to %[[VAL_4]] : !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
  ! CHECK:  %[[VAL_5:.*]] = arith.constant 1 : i32
  ! CHECK:  %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_5]] : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>, !fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK:  fir.store %[[VAL_1]]#0 to %[[VAL_6]] : !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK:  fir.call @_QFcapture_char_func_dummyPinternal(%[[VAL_2]]) {{.*}}: (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>, !fir.ref<i32>>>) -> ()
  call internal()
contains
  ! CHECK-LABEL: func.func private @_QFcapture_char_func_dummyPinternal(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>, !fir.ref<i32>>> {fir.host_assoc})
  subroutine internal()
  ! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : i32
  ! CHECK:  %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>, !fir.ref<i32>>>, i32) -> !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
  ! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
  ! CHECK:  %[[VAL_4:.*]] = arith.constant 1 : i32
  ! CHECK:  %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_4]] : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>, !fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK:  %[[VAL_6_ref:.*]] = fir.load %[[VAL_5]] : !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_6_ref]] {{.*}}uniq_name = "_QFcapture_char_func_dummyEn"
  ! CHECK:  %[[VAL_12:.*]] = fir.extract_value %[[VAL_3]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
  ! CHECK:  %[[VAL_13:.*]] = fir.box_addr %[[VAL_12]] : (!fir.boxproc<() -> ()>) -> (() -> ())
  ! CHECK:  %[[VAL_14:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
   print *, char_func_dummy()
  end subroutine
end subroutine

! CHECK-LABEL: func.func @_QPcapture_char_func_assumed_dummy(
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
! CHECK-LABEL: func.func private @_QFcapture_char_func_assumed_dummyPinternal(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>>> {fir.host_assoc})
  subroutine internal()
! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<tuple<tuple<!fir.boxproc<() -> ()>, i64>>>, i32) -> !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<tuple<!fir.boxproc<() -> ()>, i64>>
! CHECK:  %[[VAL_9:.*]] = fir.extract_value %[[VAL_3]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_10:.*]] = fir.box_addr %[[VAL_9]] : (!fir.boxproc<() -> ()>) -> (() -> ())
   print *, char_func_dummy()
  end subroutine
end subroutine

! CHECK-LABEL: func.func @_QPcapture_char_func(
! CHECK-SAME:  %[[VAL_0_arg:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
subroutine capture_char_func(n)
  character(n), external :: char_func
! CHECK:  %[[VAL_0:.*]]:2 = hlfir.declare %[[VAL_0_arg]] {{.*}}uniq_name = "_QFcapture_char_funcEn"
! CHECK:  %[[VAL_1:.*]] = fir.alloca tuple<!fir.ref<i32>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  fir.store %[[VAL_0]]#0 to %[[VAL_3]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  fir.call @_QFcapture_char_funcPinternal(%[[VAL_1]]) {{.*}}: (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
  call internal()
contains
! CHECK-LABEL: func.func private @_QFcapture_char_funcPinternal(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc})
  subroutine internal()
   ! CHECK: %[[VAL_1:.*]] = arith.constant 0 : i32
   ! CHECK: %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
   ! CHECK: %[[VAL_3_ref:.*]] = fir.load %[[VAL_2]] : !fir.llvm_ptr<!fir.ref<i32>>
   ! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_3_ref]] {{.*}}uniq_name = "_QFcapture_char_funcEn"
   print *, char_func()
  end subroutine
end subroutine

! CHECK-LABEL: func.func @_QPcapture_array_func(
! CHECK-SAME:  %[[VAL_0_arg:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
subroutine capture_array_func(n)
  integer :: n
  interface
  function array_func()
    import :: n
    integer :: array_func(n)
  end function
  end interface
! CHECK:  %[[VAL_0:.*]]:2 = hlfir.declare %[[VAL_0_arg]] {{.*}}uniq_name = "_QFcapture_array_funcEn"
! CHECK:  %[[VAL_1:.*]] = fir.alloca tuple<!fir.ref<i32>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  fir.store %[[VAL_0]]#0 to %[[VAL_3]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  fir.call @_QFcapture_array_funcPinternal(%[[VAL_1]]) {{.*}}: (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
  call internal()
contains
  subroutine internal()
! CHECK-LABEL: func.func private @_QFcapture_array_funcPinternal(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc})
! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  %[[VAL_3_ref:.*]] = fir.load %[[VAL_2]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_3_ref]] {{.*}}uniq_name = "_QFcapture_array_funcEn"
! CHECK:  %[[VAL_9:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
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

! CHECK-LABEL: func.func @_QPuse_module() {
subroutine use_module()
  ! verify there is no capture triggers by the interface.
  use define_char_func
! CHECK:  fir.call @_QFuse_modulePinternal()
  call internal()
  contains
! CHECK-LABEL: func.func private @_QFuse_modulePinternal()
  subroutine internal()
    print *, return_char(42)
  end subroutine
end subroutine
