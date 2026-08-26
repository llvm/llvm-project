! RUN: bbc -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

program character_array_constant
  character(4) :: values(10000)
  character(4) :: words(3)
  data values / 10000 * ' ' /
  data words / 3 * 'ab' /
end program

! CHECK-LABEL: fir.global internal @_QFEvalues
! CHECK: %[[UNDEF:.*]] = fir.undefined !fir.array<10000x!fir.char<1,4>>
! CHECK-NEXT: %[[SPACE:.*]] = fir.string_lit "    "(4) : !fir.char<1,4>
! CHECK-NEXT: %[[INIT:.*]] = fir.insert_on_range %[[UNDEF]], %[[SPACE]] from (0) to (9999) : (!fir.array<10000x!fir.char<1,4>>, !fir.char<1,4>) -> !fir.array<10000x!fir.char<1,4>>
! CHECK: fir.has_value %[[INIT]] : !fir.array<10000x!fir.char<1,4>>

! CHECK-LABEL: fir.global internal @_QFEwords
! CHECK: %[[UNDEF:.*]] = fir.undefined !fir.array<3x!fir.char<1,4>>
! CHECK-NEXT: %[[AB:.*]] = fir.string_lit "ab "(4) : !fir.char<1,4>
! CHECK-NEXT: %[[INIT:.*]] = fir.insert_on_range %[[UNDEF]], %[[AB]] from (0) to (2) : (!fir.array<3x!fir.char<1,4>>, !fir.char<1,4>) -> !fir.array<3x!fir.char<1,4>>
! CHECK: fir.has_value %[[INIT]] : !fir.array<3x!fir.char<1,4>>

! LLVM: @_QFEvalues = internal global [40000 x i8]
! LLVM: @_QFEwords = internal global [12 x i8] c"ab  ab  ab  "
