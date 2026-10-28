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

! LLVM: @_QFEvalues = internal global [10000 x [4 x i8]]
! LLVM: @_QFEwords = internal global [3 x [4 x i8]] [{{.*}}c"ab  ", {{.*}}c"ab  ", {{.*}}c"ab  "]

! A CHARACTER array component is an array of characters nested inside a
! structure, so its initial value cannot be described by an array attribute.
subroutine component()
  type t
    character(2) :: c(2)
  end type
  type(t), save :: x = t(c=["ab", "ab"])
  call use_x(x)
end subroutine

! CHECK-LABEL: fir.global internal @_QFcomponentEx
! CHECK: %[[AB:.*]] = fir.string_lit "ab"(2) : !fir.char<1,2>
! CHECK-NEXT: fir.insert_on_range %{{.*}}, %[[AB]] from (0) to (1) : (!fir.array<2x!fir.char<1,2>>, !fir.char<1,2>) -> !fir.array<2x!fir.char<1,2>>

! LLVM: @_QFcomponentEx = internal global %_QFcomponentTt { [2 x [2 x i8]] [{{.*}}c"ab", {{.*}}c"ab"] }
