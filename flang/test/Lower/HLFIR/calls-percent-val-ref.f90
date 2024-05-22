! Test lowering of legacy %VAL and %REF actual arguments.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test_val_1(x)
  integer :: x
  call val1(%val(x))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_val_1(
! CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_val_1Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
! CHECK:           fir.call @_QPval1(%[[VAL_2]]) fastmath<contract> : (i32) -> ()

subroutine test_val_2(x)
  complex, allocatable :: x
  call val2(%val(x))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_val_2(
! CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.complex<4>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_val_2Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.complex<4>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.complex<4>>>>, !fir.ref<!fir.box<!fir.heap<!fir.complex<4>>>>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.complex<4>>>>
! CHECK:           %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<!fir.complex<4>>>) -> !fir.heap<!fir.complex<4>>
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.heap<!fir.complex<4>>
! CHECK:           fir.call @_QPval2(%[[VAL_4]]) fastmath<contract> : (!fir.complex<4>) -> ()

subroutine test_ref_char(x)
  ! There must be not extra length argument. Only the address is
  ! passed.
  character(*) :: x
  call ref_char(%ref(x))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_ref_char(
! CHECK-SAME:                                %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]]#0 typeparams %[[VAL_1]]#1 {uniq_name = "_QFtest_ref_charEx"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_2]]#0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           fir.call @_QPref_char(%[[VAL_3]]#0) fastmath<contract> : (!fir.ref<!fir.char<1,?>>) -> ()

subroutine test_ref_1(x)
  integer :: x
  call ref1(%ref(x))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_ref_1(
! CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_ref_1Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           fir.call @_QPref1(%[[VAL_1]]#1) fastmath<contract> : (!fir.ref<i32>) -> ()

subroutine test_ref_2(x)
  complex, pointer :: x
  call ref2(%ref(x))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_ref_2(
! CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_ref_2Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
! CHECK:           %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.complex<4>>>) -> !fir.ptr<!fir.complex<4>>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ptr<!fir.complex<4>>) -> !fir.ref<!fir.complex<4>>
! CHECK:           fir.call @_QPref2(%[[VAL_4]]) fastmath<contract> : (!fir.ref<!fir.complex<4>>) -> ()

subroutine test_skip_copy_in_out(x)
  real :: x(:)
  call val3(%val(%loc(x)))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_skip_copy_in_out(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_skip_copy_in_outEx"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]]#1 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<?xf32>>) -> i64
! CHECK:           fir.call @_QPval3(%[[VAL_3]]) fastmath<contract> : (i64) -> ()
