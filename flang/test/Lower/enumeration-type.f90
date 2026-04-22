! Test lowering of enumeration types to HLFIR/FIR.
! Enumeration types lower to i32 values representing 1-based ordinal positions.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module enum_mod
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
end module

! -----------------------------------------------------------------------------
!            Test enumeration type maps to i32 (not fir.type)
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_enum_variable()
subroutine test_enum_variable()
  use enum_mod
  type(color) :: c
  ! CHECK: %[[ALLOC:.*]] = fir.alloca i32
  ! CHECK: hlfir.declare %[[ALLOC]]
  c = red
end subroutine

! -----------------------------------------------------------------------------
!            Test enumerator constants lower to i32 constants
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_enumerator_constants()
subroutine test_enumerator_constants()
  use enum_mod
  type(color) :: c
  ! CHECK: %[[RED:.*]] = arith.constant 1 : i32
  ! CHECK: hlfir.assign %[[RED]]
  c = red
  ! CHECK: %[[GREEN:.*]] = arith.constant 2 : i32
  ! CHECK: hlfir.assign %[[GREEN]]
  c = green
  ! CHECK: %[[BLUE:.*]] = arith.constant 3 : i32
  ! CHECK: hlfir.assign %[[BLUE]]
  c = blue
end subroutine

! -----------------------------------------------------------------------------
!            Test enumeration constructor — color(n) → i32 constant
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_constructor()
subroutine test_constructor()
  use enum_mod
  type(color) :: c
  ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
  ! CHECK: hlfir.assign %[[C2]]
  c = color(2)
end subroutine

! -----------------------------------------------------------------------------
!            Test enumeration comparisons (relational operators)
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_comparisons(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32>{{.*}}, %[[ARG1:.*]]: !fir.ref<i32>{{.*}})
subroutine test_comparisons(c1, c2)
  use enum_mod
  type(color), intent(in) :: c1, c2
  logical :: l
  ! CHECK: %[[V1:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK: %[[V2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK: arith.cmpi eq, %[[V1]], %[[V2]] : i32
  l = (c1 == c2)
  ! CHECK: arith.cmpi slt
  l = (c1 < c2)
  ! CHECK: arith.cmpi sle
  l = (c1 <= c2)
  ! CHECK: arith.cmpi sgt
  l = (c1 > c2)
  ! CHECK: arith.cmpi sge
  l = (c1 >= c2)
  ! CHECK: arith.cmpi ne
  l = (c1 /= c2)
end subroutine

! -----------------------------------------------------------------------------
!            Test INT() conversion of enumeration values
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_int_conversion()
subroutine test_int_conversion()
  use enum_mod
  integer :: i
  ! CHECK: %[[C1:.*]] = arith.constant 1 : i32
  i = int(red)
end subroutine

! -----------------------------------------------------------------------------
!            Test HUGE() — returns enumerator count as i32 constant
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_huge()
subroutine test_huge()
  use enum_mod
  type(color) :: c
  ! CHECK: arith.constant 3 : i32
  c = huge(red)
end subroutine

! -----------------------------------------------------------------------------
!            Test NEXT() with variable argument
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_next(
! CHECK-SAME: %[[ARG:.*]]: !fir.ref<i32>
subroutine test_next(c)
  use enum_mod
  type(color), intent(in) :: c
  type(color) :: result
  integer :: stat
  ! CHECK: %[[ORD:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! Compute: min(ordinal + 1, 3)
  ! CHECK: %[[ONE:.*]] = arith.constant 1 : i32
  ! CHECK: %[[INC:.*]] = arith.addi %[[ORD]], %[[ONE]] : i32
  ! CHECK: %[[MAX:.*]] = arith.constant 3 : i32
  ! CHECK: %[[CMP:.*]] = arith.cmpi sle, %[[INC]], %[[MAX]] : i32
  ! CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[INC]], %[[MAX]] : i32
  ! Boundary check: ordinal == 3
  ! CHECK: %[[BOUND:.*]] = arith.cmpi eq, %[[ORD]], %[[MAX]] : i32
  ! STAT handling: select 112 or 0
  ! CHECK: arith.constant 112
  ! CHECK: arith.constant 0
  ! CHECK: arith.select %[[BOUND]]
  ! CHECK: hlfir.assign
  result = next(c, stat=stat)
end subroutine

! -----------------------------------------------------------------------------
!            Test PREVIOUS() with variable argument
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_previous(
! CHECK-SAME: %[[ARG:.*]]: !fir.ref<i32>
subroutine test_previous(c)
  use enum_mod
  type(color), intent(in) :: c
  type(color) :: result
  integer :: stat
  ! CHECK: %[[ORD:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! Compute: max(ordinal - 1, 1)
  ! CHECK: %[[ONE:.*]] = arith.constant 1 : i32
  ! CHECK: %[[DEC:.*]] = arith.subi %[[ORD]], %[[ONE]] : i32
  ! CHECK: %[[CMP:.*]] = arith.cmpi sge, %[[DEC]], %[[ONE]] : i32
  ! CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[DEC]], %[[ONE]] : i32
  ! Boundary check: ordinal == 1
  ! CHECK: %[[BOUND:.*]] = arith.cmpi eq, %[[ORD]], %[[ONE]] : i32
  ! STAT handling: select 112 or 0
  ! CHECK: arith.constant 112
  ! CHECK: arith.constant 0
  ! CHECK: arith.select %[[BOUND]]
  ! CHECK: hlfir.assign
  result = previous(c, stat=stat)
end subroutine

! -----------------------------------------------------------------------------
!            Test NEXT() without STAT — generates fatal error path
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_next_no_stat(
subroutine test_next_no_stat(c)
  use enum_mod
  type(color), intent(in) :: c
  type(color) :: result
  ! CHECK: %[[ORD:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK: arith.addi
  ! CHECK: arith.cmpi sle
  ! CHECK: arith.select
  ! Boundary without STAT — fir.if for fatal error
  ! CHECK: %[[BOUND:.*]] = arith.cmpi eq
  ! CHECK: fir.if %[[BOUND]]
  ! CHECK:   fir.call @{{.*}}ReportFatalUserError
  ! CHECK: }
  result = next(c)
end subroutine

! -----------------------------------------------------------------------------
!            Test SELECT CASE with enumeration type
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_select_case(
subroutine test_select_case(c)
  use enum_mod
  type(color), intent(in) :: c
  integer :: result
  ! CHECK: %[[SEL:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK: %[[C1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
  ! CHECK: %[[C3:.*]] = arith.constant 3 : i32
  ! CHECK: fir.select_case %[[SEL]] : i32 [#fir.point, %[[C1]], ^{{.*}}, #fir.point, %[[C2]], ^{{.*}}, #fir.point, %[[C3]], ^{{.*}}, unit, ^{{.*}}]
  select case (c)
    case (red)
      result = 1
    case (green)
      result = 2
    case (blue)
      result = 3
  end select
end subroutine

! -----------------------------------------------------------------------------
!            Test enumeration dummy argument passing
! -----------------------------------------------------------------------------

! -----------------------------------------------------------------------------
!            Test formatted WRITE of enumeration value
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_formatted_write(
! CHECK-SAME: %[[ARG:.*]]: !fir.ref<i32>
subroutine test_formatted_write(c)
  use enum_mod
  type(color), intent(in) :: c
  ! CHECK: fir.call @_FortranAioBeginExternalFormattedOutput
  ! CHECK: %[[VAL:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[VAL]])
  ! CHECK: fir.call @_FortranAioEndIoStatement
  write(*, '(I4)') c
end subroutine

! -----------------------------------------------------------------------------
!            Test formatted READ into enumeration variable
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_formatted_read(
! CHECK-SAME: %[[ARG:.*]]: !fir.ref<i32>
subroutine test_formatted_read(c)
  use enum_mod
  type(color), intent(inout) :: c
  ! CHECK: fir.call @_FortranAioBeginExternalFormattedInput
  ! CHECK: %[[CONV:.*]] = fir.convert %{{.*}} : (!fir.ref<i32>) -> !fir.ref<i64>
  ! CHECK: fir.call @_FortranAioInputInteger(%{{.*}}, %[[CONV]], %{{.*}})
  ! CHECK: fir.call @_FortranAioEndIoStatement
  read(*, '(I4)') c
end subroutine

! -----------------------------------------------------------------------------
!            Test enumeration dummy argument passing
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_enum_arg_pass()
subroutine test_enum_arg_pass()
  use enum_mod
  type(color) :: c
  c = green
  ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
  ! CHECK: fir.call @_QPtake_enum
  call take_enum(c)
end subroutine

! CHECK-LABEL: func.func @_QPtake_enum(
! CHECK-SAME: %[[ARG:.*]]: !fir.ref<i32>
subroutine take_enum(c)
  use enum_mod
  type(color), intent(in) :: c
end subroutine
