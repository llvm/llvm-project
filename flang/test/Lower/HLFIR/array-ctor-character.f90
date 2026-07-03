! Test lowering of character array constructors to HLFIR.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module chararrayctor
  character(3), target :: ctarg1 = "abc"
  character(3), target :: ctarg2 = "def"
contains

  subroutine test_pre_computed_length(c1, c2)
    character(*) :: c1, c2
    call takes_char([character(3):: c1, c2])
  end subroutine
! CHECK-LABEL: func.func @_QMchararrayctorPtest_pre_computed_length(
! CHECK:  %[[VAL_9:.*]]:2 = hlfir.declare %{{.*}}Ec1"
! CHECK:  %[[VAL_11:.*]]:2 = hlfir.declare %{{.*}}Ec2"
! CHECK:  %[[VAL_12:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_13:.*]] = arith.constant 3 : i64
! CHECK:  %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:  %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_15B:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_16:.*]] = fir.allocmem !fir.array<2x!fir.char<1,3>> {bindc_name = ".tmp.arrayctor", uniq_name = ""}
! CHECK:  %[[VAL_17:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_16]](%[[VAL_17]]) typeparams %[[VAL_14]] {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<2x!fir.char<1,3>>>, !fir.shape<1>, index) -> (!fir.heap<!fir.array<2x!fir.char<1,3>>>, !fir.heap<!fir.array<2x!fir.char<1,3>>>)
! CHECK:  %[[VAL_19:.*]] = arith.constant 3 : i64
! CHECK:  %[[VAL_20:.*]] = hlfir.set_length %[[VAL_9]]#0 len %[[VAL_19]] : (!fir.boxchar<1>, i64) -> !hlfir.expr<!fir.char<1,3>>
! CHECK:  %[[VAL_21:.*]] = arith.addi %[[VAL_15]], %[[VAL_15B]] : index
! CHECK:  %[[VAL_22:.*]] = hlfir.designate %[[VAL_18]]#0 (%[[VAL_15]])  typeparams %[[VAL_14]] : (!fir.heap<!fir.array<2x!fir.char<1,3>>>, index, index) -> !fir.ref<!fir.char<1,3>>
! CHECK:  hlfir.assign %[[VAL_20]] to %[[VAL_22]] : !hlfir.expr<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>
! CHECK:  %[[VAL_23:.*]] = arith.constant 3 : i64
! CHECK:  %[[VAL_24:.*]] = hlfir.set_length %[[VAL_11]]#0 len %[[VAL_23]] : (!fir.boxchar<1>, i64) -> !hlfir.expr<!fir.char<1,3>>
! CHECK:  %[[VAL_25:.*]] = hlfir.designate %[[VAL_18]]#0 (%[[VAL_21]])  typeparams %[[VAL_14]] : (!fir.heap<!fir.array<2x!fir.char<1,3>>>, index, index) -> !fir.ref<!fir.char<1,3>>
! CHECK:  hlfir.assign %[[VAL_24]] to %[[VAL_25]] : !hlfir.expr<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>
! CHECK:  %[[VAL_26:.*]] = arith.constant true
! CHECK:  %[[VAL_27:.*]] = hlfir.as_expr %[[VAL_18]]#0 move %[[VAL_26]] : (!fir.heap<!fir.array<2x!fir.char<1,3>>>, i1) -> !hlfir.expr<2x!fir.char<1,3>>
! CHECK:  fir.call @_QMchararrayctorPtakes_char
! CHECK:  hlfir.destroy %[[VAL_27]] : !hlfir.expr<2x!fir.char<1,3>>

  subroutine test_dynamic_length()
    call takes_char([char_pointer(1), char_pointer(2)])
  end subroutine
! CHECK-LABEL: func.func @_QMchararrayctorPtest_dynamic_length() {
! CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.char<1,?>>> {bindc_name = ".result"}
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.char<1,?>>> {bindc_name = ".result"}
! CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.array<10xi64> {bindc_name = ".rt.arrayctor.vector"}
! CHECK:  %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<2x!fir.char<1,?>>>> {bindc_name = ".tmp.arrayctor"}
! CHECK:  %[[VAL_10:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_11:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_12:.*]] = fir.zero_bits !fir.heap<!fir.array<2x!fir.char<1,?>>>
! CHECK:  %[[VAL_13:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_14:.*]] = fir.embox %[[VAL_12]](%[[VAL_13]]) typeparams %[[VAL_11]] : (!fir.heap<!fir.array<2x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.heap<!fir.array<2x!fir.char<1,?>>>>
! CHECK:  fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,?>>>>>
! CHECK:  %[[VAL_15:.*]] = arith.constant true
! CHECK:  %[[VAL_16:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<10xi64>>) -> !fir.llvm_ptr<i8>
! CHECK:  %[[VAL_20:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:  fir.call @_FortranAInitArrayConstructorVector(%[[VAL_16]], %[[VAL_20]], %[[VAL_15]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.llvm_ptr<i8>, !fir.ref<!fir.box<none>>, i1, !fir.ref<i8>, i32) -> ()
! CHECK:  fir.call @_QMchararrayctorPchar_pointer(
! CHECK:  fir.call @_FortranAPushArrayConstructorValue(%[[VAL_16]], %{{.*}}) {{.*}}: (!fir.llvm_ptr<i8>, !fir.box<none>) -> ()
! CHECK:  fir.call @_QMchararrayctorPchar_pointer(
! CHECK:  fir.call @_FortranAPushArrayConstructorValue(%[[VAL_16]], %{{.*}}) {{.*}}: (!fir.llvm_ptr<i8>, !fir.box<none>) -> ()
! CHECK:  %[[VAL_45:.*]] = arith.constant true
! CHECK:  %[[VAL_46:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,?>>>>>
! CHECK:  %[[VAL_47:.*]] = hlfir.as_expr %[[VAL_46]] move %[[VAL_45]] : (!fir.box<!fir.heap<!fir.array<2x!fir.char<1,?>>>>, i1) -> !hlfir.expr<2x!fir.char<1,?>>
! CHECK:  fir.call @_QMchararrayctorPtakes_char(
! CHECK:  hlfir.destroy %[[VAL_47]] : !hlfir.expr<2x!fir.char<1,?>>


! Code below is only relevant for end-to-end test validation purpose.
  function char_pointer(i)
    integer :: i
    character(:), pointer :: char_pointer
    if (i.eq.1) then
      char_pointer => ctarg1
    else
      char_pointer => ctarg2
    end if
  end function
  subroutine takes_char(c)
    character(*) :: c(:)
    print *, "got   : ", c
  end subroutine
end module

  use chararrayctor
  print *, "expect: ab cde"
  call test_pre_computed_length("ab", "cdefg")
  print *, "expect: abcdef"
  call test_dynamic_length()
end

subroutine test_set_length_sanitize(i, c1)
  integer(8) :: i
  character(*) :: c1
  call takes_char([character(len=i):: c1])
end subroutine
! CHECK-LABEL:   func.func @_QPtest_set_length_sanitize(
! CHECK:   %[[VAL_2:.*]]:2 = hlfir.declare {{.*}}Ec1
! CHECK:   %[[VAL_3:.*]]:2 = hlfir.declare %arg0
! CHECK:   %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i64>
! CHECK:   %[[VAL_25:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i64>
! CHECK:   %[[VAL_26:.*]] = arith.constant 0 : i64
! CHECK:   %[[VAL_27:.*]] = arith.cmpi sgt, %[[VAL_25]], %[[VAL_26]] : i64
! CHECK:   %[[VAL_28:.*]] = arith.select %[[VAL_27]], %[[VAL_25]], %[[VAL_26]] : i64
! CHECK:   %[[VAL_29:.*]] = hlfir.set_length %[[VAL_2]]#0 len %[[VAL_28]] : (!fir.boxchar<1>, i64) -> !hlfir.expr<!fir.char<1,?>>
