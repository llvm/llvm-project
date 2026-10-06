! Test lowering of enumeration types to HLFIR.
! An enumeration type lowers to a record type with a single i32 component,
! __ordinal, holding the 1-based ordinal of the enumerator.
! RUN: %flang_fc1 -fenumeration-type -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fenumeration-type -emit-fir %s -o /dev/null
! RUN: %flang_fc1 -fenumeration-type -emit-fir -mmlir -strict-fir-volatile-verifier %s -o /dev/null

module enum_mod
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
end module

! -----------------------------------------------------------------------------
!            Test enumeration variable is a record type
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_enum_variable()
subroutine test_enum_variable()
  use enum_mod
  type(color) :: c
  ! CHECK: %[[ALLOC:.*]] = fir.alloca !fir.type<_QMenum_modTcolor{__ordinal:i32}> <{bindc_name = "c"
  ! CHECK: hlfir.declare %[[ALLOC]]
  c = red
end subroutine

! -----------------------------------------------------------------------------
!            Test enumerator constants lower to record constants
! -----------------------------------------------------------------------------

! The ordinal of each read-only constant is checked with the globals at the end
! of the file.

! CHECK-LABEL: func.func @_QPtest_enumerator_constants()
subroutine test_enumerator_constants()
  use enum_mod
  type(color) :: c
  ! CHECK: %[[C:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_enumerator_constantsEc"}
  ! CHECK: %[[RED_ADDR:.*]] = fir.address_of(@[[RED:_QQro\._QMenum_modTcolor\.[0-9]+]])
  ! CHECK: %[[RED_DECL:.*]]:2 = hlfir.declare %[[RED_ADDR]]
  ! CHECK: hlfir.assign %[[RED_DECL]]#0 to %[[C]]#0 : !fir.ref<!fir.type<_QMenum_modTcolor{__ordinal:i32}>>, !fir.ref<!fir.type<_QMenum_modTcolor{__ordinal:i32}>>
  c = red
  ! CHECK: fir.address_of(@[[GREEN:_QQro\._QMenum_modTcolor\.[0-9]+]])
  ! CHECK: hlfir.assign
  c = green
  ! CHECK: fir.address_of(@[[BLUE:_QQro\._QMenum_modTcolor\.[0-9]+]])
  ! CHECK: hlfir.assign
  c = blue
end subroutine

! -----------------------------------------------------------------------------
!            Test enumeration constructor with a constant argument
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_constructor()
subroutine test_constructor()
  use enum_mod
  type(color) :: c
  ! CHECK: fir.address_of(@[[CTOR2:_QQro\._QMenum_modTcolor\.[0-9]+]])
  ! CHECK: hlfir.assign
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
! CHECK-SAME: %{{.*}}: !fir.ref<i32>
subroutine test_constructor_runtime(i)
  use enum_mod
  integer, intent(in) :: i
  type(color) :: c
  ! CHECK: %[[C:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_constructor_runtimeEc"}
  ! CHECK: %[[ORD:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
  ! CHECK-DAG: %[[MAX:.*]] = arith.constant 3 : i32
  ! CHECK: %[[LOW:.*]] = arith.cmpi slt, %[[ORD]], %[[ONE]] : i32
  ! CHECK: %[[HIGH:.*]] = arith.cmpi sgt, %[[ORD]], %[[MAX]] : i32
  ! CHECK: %[[OOR:.*]] = arith.ori %[[LOW]], %[[HIGH]] : i1
  ! CHECK: fir.if %[[OOR]] {
  ! CHECK:   fir.call @{{.*}}ReportFatalUserError
  ! CHECK: }
  ! CHECK: %[[TMP:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "ctor.temp"}
  ! CHECK: %[[F:.*]] = hlfir.designate %[[TMP]]#0{"__ordinal"}
  ! CHECK: hlfir.assign %[[ORD]] to %[[F]] : i32, !fir.ref<i32>
  ! CHECK: hlfir.assign %[[TMP]]#0 to %[[C]]#0
  c = color(i)
end subroutine

! The range check uses the argument's own kind, before it is narrowed to the
! i32 ordinal, so a large INTEGER(8) value cannot wrap into range.

! CHECK-LABEL: func.func @_QPtest_constructor_int8(
! CHECK-SAME: %{{.*}}: !fir.ref<i64>
subroutine test_constructor_int8(i)
  use enum_mod
  integer(8), intent(in) :: i
  type(color) :: c
  ! CHECK: %[[I:.*]] = fir.load %{{.*}} : !fir.ref<i64>
  ! CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i64
  ! CHECK-DAG: %[[MAX:.*]] = arith.constant 3 : i64
  ! CHECK: %[[LOW:.*]] = arith.cmpi slt, %[[I]], %[[ONE]] : i64
  ! CHECK: %[[HIGH:.*]] = arith.cmpi sgt, %[[I]], %[[MAX]] : i64
  ! CHECK: %[[OOR:.*]] = arith.ori %[[LOW]], %[[HIGH]] : i1
  ! CHECK: fir.if %[[OOR]] {
  ! CHECK:   fir.call @{{.*}}ReportFatalUserError
  ! CHECK: }
  ! CHECK: %[[ORD:.*]] = fir.convert %[[I]] : (i64) -> i32
  ! CHECK: %[[TMP:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "ctor.temp"}
  ! CHECK: %[[F:.*]] = hlfir.designate %[[TMP]]#0{"__ordinal"}
  ! CHECK: hlfir.assign %[[ORD]] to %[[F]] : i32, !fir.ref<i32>
  c = color(i)
end subroutine

! A narrower argument is widened to i32 first so the enumerator count cannot
! wrap in the argument's kind.

! CHECK-LABEL: func.func @_QPtest_constructor_int1(
! CHECK-SAME: %{{.*}}: !fir.ref<i8>
subroutine test_constructor_int1(i)
  use enum_mod
  integer(1), intent(in) :: i
  type(color) :: c
  ! CHECK: %[[I:.*]] = fir.load %{{.*}} : !fir.ref<i8>
  ! CHECK: %[[ORD:.*]] = fir.convert %[[I]] : (i8) -> i32
  ! CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
  ! CHECK-DAG: %[[MAX:.*]] = arith.constant 3 : i32
  ! CHECK: %[[LOW:.*]] = arith.cmpi slt, %[[ORD]], %[[ONE]] : i32
  ! CHECK: %[[HIGH:.*]] = arith.cmpi sgt, %[[ORD]], %[[MAX]] : i32
  ! CHECK: %[[OOR:.*]] = arith.ori %[[LOW]], %[[HIGH]] : i1
  ! CHECK: fir.if %[[OOR]] {
  ! CHECK:   fir.call @{{.*}}ReportFatalUserError
  ! CHECK: }
  ! CHECK: %[[TMP:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "ctor.temp"}
  ! CHECK: %[[F:.*]] = hlfir.designate %[[TMP]]#0{"__ordinal"}
  ! CHECK: hlfir.assign %[[ORD]] to %[[F]] : i32, !fir.ref<i32>
  c = color(i)
end subroutine

! -----------------------------------------------------------------------------
!            Test enumeration comparisons (relational operators)
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_comparisons(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.type<_QMenum_modTcolor{__ordinal:i32}>> {fir.bindc_name = "c1"}, %{{.*}}: !fir.ref<!fir.type<_QMenum_modTcolor{__ordinal:i32}>> {fir.bindc_name = "c2"})
subroutine test_comparisons(c1, c2)
  use enum_mod
  type(color), intent(in) :: c1, c2
  logical :: l
  ! CHECK: %[[C1:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_comparisonsEc1"}
  ! CHECK: %[[C2:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_comparisonsEc2"}
  ! CHECK: %[[F1:.*]] = hlfir.designate %[[C1]]#0{"__ordinal"}
  ! CHECK: %[[V1:.*]] = fir.load %[[F1]] : !fir.ref<i32>
  ! CHECK: %[[F2:.*]] = hlfir.designate %[[C2]]#0{"__ordinal"}
  ! CHECK: %[[V2:.*]] = fir.load %[[F2]] : !fir.ref<i32>
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
  ! CHECK: %[[I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_int_conversionEi"}
  ! CHECK: %[[C1:.*]] = arith.constant 1 : i32
  ! CHECK: hlfir.assign %[[C1]] to %[[I]]#0 : i32, !fir.ref<i32>
  i = int(red)
end subroutine

! CHECK-LABEL: func.func @_QPtest_int_variable(
subroutine test_int_variable(c, arr)
  use enum_mod
  type(color), intent(in) :: c, arr(3)
  integer :: i, iarr(3)
  integer(8) :: j
  ! CHECK: %[[ARR:.*]]:2 = hlfir.declare %{{.*}}(%[[SHAPE:[0-9]+]]) dummy_scope %{{.*}} {{.*}}uniq_name = "_QFtest_int_variableEarr"}
  ! CHECK: %[[C:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_int_variableEc"}
  ! CHECK: %[[I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_int_variableEi"}
  ! CHECK: %[[IARR:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_int_variableEiarr"}
  ! CHECK: %[[J:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_int_variableEj"}
  ! CHECK: %[[F1:.*]] = hlfir.designate %[[C]]#0{"__ordinal"}
  ! CHECK: %[[V1:.*]] = fir.load %[[F1]] : !fir.ref<i32>
  ! CHECK: hlfir.assign %[[V1]] to %[[I]]#0 : i32, !fir.ref<i32>
  i = int(c)
  ! CHECK: %[[F2:.*]] = hlfir.designate %[[C]]#0{"__ordinal"}
  ! CHECK: %[[V2:.*]] = fir.load %[[F2]] : !fir.ref<i32>
  ! CHECK: %[[V2_8:.*]] = fir.convert %[[V2]] : (i32) -> i64
  ! CHECK: hlfir.assign %[[V2_8]] to %[[J]]#0 : i64, !fir.ref<i64>
  j = int(c, kind=8)
  ! CHECK: %[[FA:.*]] = hlfir.designate %[[ARR]]#0{"__ordinal"} shape %[[SHAPE]] : {{.*}} -> !fir.box<!fir.array<3xi32>>
  ! CHECK: %[[EL:.*]] = hlfir.elemental %[[SHAPE]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xi32> {
  ! CHECK: hlfir.designate %[[FA]] (%{{.*}})
  ! CHECK: hlfir.yield_element %{{.*}} : i32
  ! CHECK: hlfir.assign %[[EL]] to %[[IARR]]#0
  iarr = int(arr)
end subroutine

! The __ordinal designator keeps the VOLATILE qualification of its base.

! CHECK-LABEL: func.func @_QPtest_int_volatile(
subroutine test_int_volatile(c, arr)
  use enum_mod
  type(color), volatile :: c, arr(3)
  integer :: i, iarr(3)
  ! CHECK: %[[ARR:.*]]:2 = hlfir.declare %{{.*}}(%[[SHAPE:[0-9]+]]) dummy_scope %{{.*}} {{.*}}uniq_name = "_QFtest_int_volatileEarr"}
  ! CHECK: %[[C:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_int_volatileEc"}
  ! CHECK: %[[F:.*]] = hlfir.designate %[[C]]#0{"__ordinal"} {{.*}} -> !fir.ref<i32, volatile>
  ! CHECK: fir.load %[[F]] : !fir.ref<i32, volatile>
  i = int(c)
  ! CHECK: %[[FA:.*]] = hlfir.designate %[[ARR]]#0{"__ordinal"} shape %[[SHAPE]] : {{.*}} -> !fir.box<!fir.array<3xi32>, volatile>
  ! CHECK: hlfir.elemental %[[SHAPE]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xi32> {
  ! CHECK: %[[E:.*]] = hlfir.designate %[[FA]] (%{{.*}}) {{.*}} -> !fir.ref<i32, volatile>
  ! CHECK: fir.load %[[E]] : !fir.ref<i32, volatile>
  iarr = int(arr)
end subroutine

! -----------------------------------------------------------------------------
!            Test HUGE() — returns the last enumerator
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_huge()
subroutine test_huge()
  use enum_mod
  type(color) :: c
  ! CHECK: fir.address_of(@[[BLUE]])
  ! CHECK: hlfir.assign
  c = huge(red)
end subroutine

! -----------------------------------------------------------------------------
!            Test SELECT CASE with enumeration type
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_select_case(
subroutine test_select_case(c)
  use enum_mod
  type(color), intent(in) :: c
  integer :: result
  ! CHECK: %[[C:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_select_caseEc"}
  ! CHECK: %[[F:.*]] = hlfir.designate %[[C]]#0{"__ordinal"}
  ! CHECK: %[[SEL:.*]] = fir.load %[[F]] : !fir.ref<i32>
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
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.type<_QMenum_modTcolor{__ordinal:i32}>>
subroutine test_formatted_write(c)
  use enum_mod
  type(color), intent(in) :: c
  ! CHECK: %[[C:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_formatted_writeEc"}
  ! CHECK: fir.call @_FortranAioBeginExternalFormattedOutput
  ! CHECK: %[[BOX:.*]] = fir.embox %[[C]]#0 : (!fir.ref<!fir.type<_QMenum_modTcolor{__ordinal:i32}>>) -> !fir.box<!fir.type<_QMenum_modTcolor{__ordinal:i32}>>
  ! CHECK: %[[ARG:.*]] = fir.convert %[[BOX]]
  ! CHECK: fir.call @_FortranAioOutputDerivedType(%{{.*}}, %[[ARG]], %{{.*}})
  ! CHECK: fir.call @_FortranAioEndIoStatement
  write(*, '(I4)') c
end subroutine

! -----------------------------------------------------------------------------
!            Test formatted READ into enumeration variable
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_formatted_read(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.type<_QMenum_modTcolor{__ordinal:i32}>>
subroutine test_formatted_read(c)
  use enum_mod
  type(color), intent(inout) :: c
  ! CHECK: %[[C:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_formatted_readEc"}
  ! CHECK: fir.call @_FortranAioBeginExternalFormattedInput
  ! CHECK: %[[BOX:.*]] = fir.embox %[[C]]#0 : (!fir.ref<!fir.type<_QMenum_modTcolor{__ordinal:i32}>>) -> !fir.box<!fir.type<_QMenum_modTcolor{__ordinal:i32}>>
  ! CHECK: %[[ARG:.*]] = fir.convert %[[BOX]]
  ! CHECK: fir.call @_FortranAioInputDerivedType(%{{.*}}, %[[ARG]], %{{.*}})
  ! CHECK: fir.call @_FortranAioEndIoStatement
  read(*, '(I4)') c
end subroutine

! -----------------------------------------------------------------------------
!            Test enumeration type as a function result
! -----------------------------------------------------------------------------

module enum_func_mod
  enumeration type :: color2
    enumerator :: c2red, c2green, c2blue
  end enumeration type
contains
  ! CHECK-LABEL: func.func @_QMenum_func_modPpick() -> !fir.type<_QMenum_func_modTcolor2{__ordinal:i32}>
  function pick() result(c)
    type(color2) :: c
    c = c2blue
  end function
  ! CHECK-LABEL: func.func @_QMenum_func_modPpick_array() -> !fir.array<3x!fir.type<_QMenum_func_modTcolor2{__ordinal:i32}>>
  function pick_array() result(c)
    type(color2) :: c(3)
    c = [c2red, c2green, c2blue]
  end function
  ! CHECK-LABEL: func.func @_QMenum_func_modPpick_alloc() -> !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMenum_func_modTcolor2{__ordinal:i32}>>>>
  function pick_alloc() result(c)
    type(color2), allocatable :: c(:)
    c = [c2red, c2green, c2blue]
  end function
end module

! CHECK-LABEL: func.func @_QPtest_func_result()
subroutine test_func_result()
  use enum_func_mod
  type(color2) :: c
  logical :: l
  ! CHECK: %[[C:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_func_resultEc"}
  ! CHECK: %[[TMP:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp.func_result"}
  ! CHECK: %[[RES:.*]] = fir.call @_QMenum_func_modPpick() {{.*}}: () -> !fir.type<_QMenum_func_modTcolor2{__ordinal:i32}>
  ! CHECK: fir.save_result %[[RES]] to %[[TMP]]#0
  ! CHECK: %[[E:.*]] = hlfir.as_expr %[[TMP]]#0
  ! CHECK: hlfir.assign %[[E]] to %[[C]]#0
  c = pick()
  ! CHECK: %[[F:.*]] = hlfir.designate %[[C]]#0{"__ordinal"}
  ! CHECK: %[[V:.*]] = fir.load %[[F]] : !fir.ref<i32>
  ! CHECK: %[[THREE:.*]] = arith.constant 3 : i32
  ! CHECK: arith.cmpi eq, %[[V]], %[[THREE]] : i32
  l = (c == c2blue)
end subroutine

! CHECK-LABEL: func.func @_QPtest_func_result_array()
subroutine test_func_result_array()
  use enum_func_mod
  type(color2) :: c(3)
  ! CHECK: hlfir.eval_in_mem {{.*}} -> !hlfir.expr<3x!fir.type<_QMenum_func_modTcolor2{__ordinal:i32}>> {
  ! CHECK: ^bb0(%[[TMP:.*]]: !fir.ref<!fir.array<3x!fir.type<_QMenum_func_modTcolor2{__ordinal:i32}>>>):
  ! CHECK: %[[RES:.*]] = fir.call @_QMenum_func_modPpick_array() {{.*}}: () -> !fir.array<3x!fir.type<_QMenum_func_modTcolor2{__ordinal:i32}>>
  ! CHECK: fir.save_result %[[RES]] to %[[TMP]]
  c = pick_array()
end subroutine

! CHECK-LABEL: func.func @_QPtest_func_result_alloc()
subroutine test_func_result_alloc()
  use enum_func_mod
  type(color2), allocatable :: c(:)
  ! CHECK: %[[RES:.*]] = fir.call @_QMenum_func_modPpick_alloc() {{.*}}: () -> !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMenum_func_modTcolor2{__ordinal:i32}>>>>
  ! CHECK: fir.save_result %[[RES]] to %{{.*}} : !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMenum_func_modTcolor2{__ordinal:i32}>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMenum_func_modTcolor2{__ordinal:i32}>>>>>
  c = pick_alloc()
end subroutine

! -----------------------------------------------------------------------------
!            Test enumeration dummy argument passing
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_enum_arg_pass()
subroutine test_enum_arg_pass()
  use enum_mod
  type(color) :: c
  ! CHECK: %[[C:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_enum_arg_passEc"}
  ! CHECK: hlfir.assign %{{.*}} to %[[C]]#0
  ! CHECK: fir.call @_QPtake_enum(%[[C]]#0) {{.*}}: (!fir.ref<!fir.type<_QMenum_modTcolor{__ordinal:i32}>>) -> ()
  c = green
  call take_enum(c)
end subroutine

! CHECK-LABEL: func.func @_QPtake_enum(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.type<_QMenum_modTcolor{__ordinal:i32}>> {fir.bindc_name = "c"}
subroutine take_enum(c)
  use enum_mod
  type(color), intent(in) :: c
end subroutine

! -----------------------------------------------------------------------------
!            Test enumeration-typed scalar PARAMETER
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_enum_parameter()
subroutine test_enum_parameter()
  use enum_mod
  type(color), parameter :: cRed = red
  type(color) :: c
  ! CHECK: hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFtest_enum_parameterECcred"} : (!fir.ref<!fir.type<_QMenum_modTcolor{__ordinal:i32}>>)
  ! CHECK: fir.address_of(@[[RED]])
  ! CHECK: hlfir.assign
  c = cRed
end subroutine

! -----------------------------------------------------------------------------
!            Test enumeration array constructor
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_array_constructor()
subroutine test_array_constructor()
  use enum_mod
  type(color) :: arr(3)
  ! CHECK: %[[RO:.*]] = fir.address_of(@[[ARR:_QQro\.3x_QMenum_modTcolor\.[0-9]+]]) : !fir.ref<!fir.array<3x!fir.type<_QMenum_modTcolor{__ordinal:i32}>>>
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
  ! CHECK: hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFtest_array_parameterECpal"} : (!fir.ref<!fir.array<3x!fir.type<_QMenum_modTcolor{__ordinal:i32}>>>, !fir.shape<1>)
  ! CHECK: hlfir.assign
  arr = pal
end subroutine

! -----------------------------------------------------------------------------
!            Test SELECT TYPE and ALLOCATE with an enumeration type
! -----------------------------------------------------------------------------

! An enumeration type is a distinct dynamic type: TYPE IS (color) and
! TYPE IS (integer) must be separate guards.

! CHECK-LABEL: func.func @_QPtest_select_type(
subroutine test_select_type(x)
  use enum_mod
  class(*), intent(in) :: x
  integer :: r
  ! CHECK: fir.select_type %{{.*}} : !fir.class<none> [#fir.type_is<!fir.type<_QMenum_modTcolor{__ordinal:i32}>>, ^{{.*}}, #fir.type_is<i32>, ^{{.*}}, unit, ^{{.*}}]
  ! CHECK: fir.box_addr %{{.*}} : (!fir.class<none>) -> !fir.ref<!fir.type<_QMenum_modTcolor{__ordinal:i32}>>
  ! CHECK: fir.box_addr %{{.*}} : (!fir.class<none>) -> !fir.ref<i32>
  select type (x)
  type is (color)
    r = 1
  type is (integer)
    r = 2
  end select
end subroutine

! CHECK-LABEL: func.func @_QPtest_allocate_color()
subroutine test_allocate_color()
  use enum_mod
  class(*), allocatable :: x
  ! CHECK: %[[TD:.*]] = fir.type_desc !fir.type<_QMenum_modTcolor{__ordinal:i32}>
  ! CHECK: %[[TDARG:.*]] = fir.convert %[[TD]]
  ! CHECK: fir.call @_FortranAAllocatableInitDerivedForAllocate(%{{.*}}, %[[TDARG]], %{{.*}}, %{{.*}})
  ! CHECK: fir.call @_FortranAAllocatableAllocate(
  allocate(color :: x)
end subroutine

! -----------------------------------------------------------------------------
!            Verify the enumeration globals
! -----------------------------------------------------------------------------

! CHECK: fir.global linkonce_odr @_QMenum_modECred constant : !fir.type<_QMenum_modTcolor{__ordinal:i32}> {
! CHECK: arith.constant 1 : i32
! CHECK-NEXT: fir.insert_value
! CHECK-NEXT: fir.has_value

! The runtime type descriptor for the enumeration type.
! CHECK: fir.global linkonce_odr @_QMenum_modE.dt.color constant target : !fir.type<_QM__fortran_type_infoTderivedtype

! CHECK: fir.global internal @[[RED]] constant : !fir.type<_QMenum_modTcolor{__ordinal:i32}> {
! CHECK: arith.constant 1 : i32
! CHECK-NEXT: fir.insert_value
! CHECK-NEXT: fir.has_value
! CHECK: fir.global internal @[[GREEN]] constant : !fir.type<_QMenum_modTcolor{__ordinal:i32}> {
! CHECK: arith.constant 2 : i32
! CHECK-NEXT: fir.insert_value
! CHECK-NEXT: fir.has_value
! CHECK: fir.global internal @[[BLUE]] constant : !fir.type<_QMenum_modTcolor{__ordinal:i32}> {
! CHECK: arith.constant 3 : i32
! CHECK-NEXT: fir.insert_value
! CHECK-NEXT: fir.has_value
! CHECK: fir.global internal @[[CTOR2]] constant : !fir.type<_QMenum_modTcolor{__ordinal:i32}> {
! CHECK: arith.constant 2 : i32
! CHECK-NEXT: fir.insert_value
! CHECK-NEXT: fir.has_value

! CHECK: fir.global internal @[[ARR]] {{.*}}constant : !fir.array<3x!fir.type<_QMenum_modTcolor{__ordinal:i32}>> {
! CHECK-NEXT: %[[A0:.*]] = fir.undefined !fir.array<3x!fir.type<_QMenum_modTcolor{__ordinal:i32}>>
! CHECK-NEXT: fir.undefined !fir.type<_QMenum_modTcolor{__ordinal:i32}>
! CHECK-NEXT: fir.field_index __ordinal
! CHECK-NEXT: %[[O1:.*]] = arith.constant 1 : i32
! CHECK-NEXT: %[[E1:.*]] = fir.insert_value %{{.*}}, %[[O1]], ["__ordinal"
! CHECK-NEXT: %[[A1:.*]] = fir.insert_value %[[A0]], %[[E1]], [0 : index]
! CHECK-NEXT: fir.undefined !fir.type<_QMenum_modTcolor{__ordinal:i32}>
! CHECK-NEXT: fir.field_index __ordinal
! CHECK-NEXT: %[[O2:.*]] = arith.constant 2 : i32
! CHECK-NEXT: %[[E2:.*]] = fir.insert_value %{{.*}}, %[[O2]], ["__ordinal"
! CHECK-NEXT: %[[A2:.*]] = fir.insert_value %[[A1]], %[[E2]], [1 : index]
! CHECK-NEXT: fir.undefined !fir.type<_QMenum_modTcolor{__ordinal:i32}>
! CHECK-NEXT: fir.field_index __ordinal
! CHECK-NEXT: %[[O3:.*]] = arith.constant 3 : i32
! CHECK-NEXT: %[[E3:.*]] = fir.insert_value %{{.*}}, %[[O3]], ["__ordinal"
! CHECK-NEXT: %[[A3:.*]] = fir.insert_value %[[A2]], %[[E3]], [2 : index]
! CHECK-NEXT: fir.has_value %[[A3]]
