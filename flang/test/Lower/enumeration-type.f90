! Test lowering of enumeration types to HLFIR/FIR.
! Enumeration types lower to i32 values representing 1-based ordinal positions.
! RUN: %flang_fc1 -fenumeration-type -emit-hlfir %s -o - | FileCheck %s

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
  ! Constant argument is range-checked at compile time (semantics), so no
  ! runtime range check is emitted here.
  ! CHECK-NOT: fir.call @{{.*}}ReportFatalUserError
  c = color(2)
end subroutine

! -----------------------------------------------------------------------------
!            Test enumeration constructor — color(i) runtime range check
! -----------------------------------------------------------------------------

! A non-constant argument cannot be range-checked at compile time, so lowering
! emits an always-on runtime check (1 <= i <= enumeratorCount) with fatal
! error termination (F2023 7.6.2 para 5).

! CHECK-LABEL: func.func @_QPtest_constructor_runtime(
! CHECK-SAME: %[[ARG:.*]]: !fir.ref<i32>
subroutine test_constructor_runtime(i)
  use enum_mod
  integer, intent(in) :: i
  type(color) :: c
  ! CHECK: %[[ORD:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
  ! CHECK-DAG: %[[MAX:.*]] = arith.constant 3 : i32
  ! CHECK: %[[LOW:.*]] = arith.cmpi slt, %[[ORD]], %[[ONE]] : i32
  ! CHECK: %[[HIGH:.*]] = arith.cmpi sgt, %[[ORD]], %[[MAX]] : i32
  ! CHECK: %[[OOR:.*]] = arith.ori %[[LOW]], %[[HIGH]] : i1
  ! CHECK: fir.if %[[OOR]] {
  ! CHECK:   fir.call @{{.*}}ReportFatalUserError
  ! CHECK: }
  ! CHECK: hlfir.assign %[[ORD]]
  c = color(i)
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
  ! Compute: min(ordinal + 1, 3). Constants are hoisted, so match order-free.
  ! CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
  ! CHECK-DAG: %[[MAX:.*]] = arith.constant 3 : i32
  ! CHECK: %[[INC:.*]] = arith.addi %[[ORD]], %[[ONE]] : i32
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
!            Test enumeration type as a function result
! -----------------------------------------------------------------------------

! An enumeration result lowers to i32 and is returned by value like an integer;
! it must not use the caller-allocated fir.save_result ABI reserved for
! record-shaped derived results.

module enum_func_mod
  enumeration type :: color2
    enumerator :: c2red, c2green, c2blue
  end enumeration type
contains
  ! CHECK-LABEL: func.func @_QMenum_func_modPpick() -> i32
  function pick() result(c)
    type(color2) :: c
    c = c2blue
  end function
end module

! CHECK-LABEL: func.func @_QPtest_func_result()
subroutine test_func_result()
  use enum_func_mod
  type(color2) :: c
  logical :: l
  ! Result returned by value as i32, with no fir.save_result.
  ! CHECK: %[[RES:.*]] = fir.call @_QMenum_func_modPpick() {{.*}}: () -> i32
  ! CHECK-NOT: fir.save_result
  ! CHECK: hlfir.assign %[[RES]]
  c = pick()
  ! The result is a genuine enumeration value: comparison lowers to i32 cmpi.
  ! CHECK: arith.cmpi eq, %{{.*}}, %{{.*}} : i32
  l = (c == c2blue)
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

! -----------------------------------------------------------------------------
!            Test enumeration-typed scalar PARAMETER
! -----------------------------------------------------------------------------

! A named constant of enumeration type must lower to an i32 constant, not a
! record type (previously asserted on cast<fir::RecordType> in ConvertConstant).

! CHECK-LABEL: func.func @_QPtest_enum_parameter()
subroutine test_enum_parameter()
  use enum_mod
  type(color), parameter :: cRed = red
  type(color) :: c
  ! CHECK: hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFtest_enum_parameterECcred"} : (!fir.ref<i32>)
  ! CHECK: %[[C1:.*]] = arith.constant 1 : i32
  ! CHECK: hlfir.assign %[[C1]]
  c = cRed
end subroutine

! -----------------------------------------------------------------------------
!            Test enumeration array constructor
! -----------------------------------------------------------------------------

! An array constructor of enumerators must lower to an i32 array constant, not a
! record-typed array (previously asserted on cast<fir::RecordType>).

! CHECK-LABEL: func.func @_QPtest_array_constructor()
subroutine test_array_constructor()
  use enum_mod
  type(color) :: arr(3)
  ! CHECK: %[[RO:.*]] = fir.address_of(@_QQro.3x_QMenum_modTcolor.{{[0-9]+}}) : !fir.ref<!fir.array<3xi32>>
  ! CHECK: hlfir.declare %[[RO]]
  ! CHECK: hlfir.assign
  arr = [red, green, blue]
end subroutine

! -----------------------------------------------------------------------------
!            Test enumeration array PARAMETER
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_array_parameter()
subroutine test_array_parameter()
  use enum_mod
  type(color), parameter :: pal(3) = [red, green, blue]
  type(color) :: arr(3)
  ! CHECK: hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFtest_array_parameterECpal"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>)
  ! CHECK: hlfir.assign
  arr = pal
end subroutine

! -----------------------------------------------------------------------------
!            Test NEXT() over a whole array (elemental)
! -----------------------------------------------------------------------------

! NEXT()/PREVIOUS() applied to an array argument lower to an hlfir.elemental over
! i32 ordinals (previously asserted on getIntOrFloatBitWidth for the array case).

! CHECK-LABEL: func.func @_QPtest_next_array(
subroutine test_next_array(arr)
  use enum_mod
  type(color), intent(in) :: arr(3)
  type(color) :: narr(3)
  integer :: stat(3)
  ! Value elemental: min(ordinal + 1, 3).
  ! CHECK: hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<3xi32> {
  ! CHECK: %[[ELE:.*]] = hlfir.designate %{{.*}} : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
  ! CHECK: %[[ORD:.*]] = fir.load %[[ELE]] : !fir.ref<i32>
  ! CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
  ! CHECK-DAG: %[[MAX:.*]] = arith.constant 3 : i32
  ! CHECK: %[[INC:.*]] = arith.addi %[[ORD]], %[[ONE]] : i32
  ! CHECK: %[[CMP:.*]] = arith.cmpi sle, %[[INC]], %[[MAX]] : i32
  ! CHECK: %[[SEL:.*]] = arith.select %[[CMP]], %[[INC]], %[[MAX]] : i32
  ! CHECK: hlfir.yield_element %[[SEL]] : i32
  ! STAT elemental: 112 at the last enumerator, else 0.
  ! CHECK: hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<3xi32> {
  ! CHECK: arith.cmpi eq, %{{.*}}, %{{.*}} : i32
  ! CHECK-DAG: arith.constant 112 : i32
  ! CHECK-DAG: arith.constant 0 : i32
  ! CHECK: arith.select
  ! CHECK: hlfir.yield_element
  narr = next(arr, stat=stat)
end subroutine

! -----------------------------------------------------------------------------
!            Test PREVIOUS() over a whole array (elemental)
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_previous_array(
subroutine test_previous_array(arr)
  use enum_mod
  type(color), intent(in) :: arr(3)
  type(color) :: parr(3)
  integer :: stat(3)
  ! Value elemental: max(ordinal - 1, 1).
  ! CHECK: hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<3xi32> {
  ! CHECK: %[[ELE:.*]] = hlfir.designate %{{.*}} : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
  ! CHECK: %[[ORD:.*]] = fir.load %[[ELE]] : !fir.ref<i32>
  ! CHECK: %[[ONE:.*]] = arith.constant 1 : i32
  ! CHECK: %[[DEC:.*]] = arith.subi %[[ORD]], %[[ONE]] : i32
  ! CHECK: %[[CMP:.*]] = arith.cmpi sge, %[[DEC]], %[[ONE]] : i32
  ! CHECK: %[[SEL:.*]] = arith.select %[[CMP]], %[[DEC]], %[[ONE]] : i32
  ! CHECK: hlfir.yield_element %[[SEL]] : i32
  parr = previous(arr, stat=stat)
end subroutine

! -----------------------------------------------------------------------------
!            Verify the enum array constructor constant is i32 ordinals 1,2,3
! -----------------------------------------------------------------------------

! CHECK: fir.global internal @_QQro.3x_QMenum_modTcolor.{{[0-9]+}} {{.*}}constant : !fir.array<3xi32> {
! CHECK: %[[G1:.*]] = arith.constant 1 : i32
! CHECK: fir.insert_value %{{.*}}, %[[G1]], [0 : index]
! CHECK: %[[G2:.*]] = arith.constant 2 : i32
! CHECK: fir.insert_value %{{.*}}, %[[G2]], [1 : index]
! CHECK: %[[G3:.*]] = arith.constant 3 : i32
! CHECK: fir.insert_value %{{.*}}, %[[G3]], [2 : index]
