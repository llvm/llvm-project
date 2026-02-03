! RUN: bbc -o - -emit-hlfir %s | FileCheck %s

! Test lowering of operations sub-expression inside elemental call arguments.
! This tests array contexts where an address is needed for each element (for
! the argument), but part of the array sub-expression must be lowered by value
! (for the operation)

module test_ops
  interface
    integer elemental function elem_func(i)
      integer, intent(in) :: i
    end function
    integer elemental function elem_func_logical(l)
      logical(8), intent(in) :: l
    end function
    integer elemental function elem_func_logical4(l)
      logical, intent(in) :: l
    end function
    integer elemental function elem_func_real(x)
      real(8), value :: x
    end function
  end interface
  integer :: i(10), j(10), iscalar
  logical(8) :: a(10), b(10)
  real(8) :: x(10), y(10)
  complex(8) :: z1(10), z2

contains
! CHECK-LABEL: func.func @_QMtest_opsPcheck_binary_ops() {
subroutine check_binary_ops()
  print *,  elem_func(i+j)
! CHECK:  %[[VAL_29:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_38:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_39:.*]] = fir.load %[[VAL_37]] : !fir.ref<i32>
! CHECK:    %[[VAL_40:.*]] = fir.load %[[VAL_38]] : !fir.ref<i32>
! CHECK:    %[[VAL_41:.*]] = arith.addi %[[VAL_39]], %[[VAL_40]] : i32
! CHECK:    hlfir.yield_element %[[VAL_41]] : i32
! CHECK:  }
! CHECK:  %[[VAL_30:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.apply %[[VAL_29]], %[[ARG0]] : (!hlfir.expr<10xi32>, index) -> i32
! CHECK:    %[[VAL_38:.*]]:3 = hlfir.associate %[[VAL_37]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:    %[[VAL_39:.*]] = fir.call @_QPelem_func(%[[VAL_38]]#0) {{.*}} : (!fir.ref<i32>) -> i32
! CHECK:    hlfir.end_associate %[[VAL_38]]#1, %[[VAL_38]]#2 : !fir.ref<i32>, i1
! CHECK:    hlfir.yield_element %[[VAL_39]] : i32
! CHECK:  }
end subroutine

! CHECK-LABEL: func.func @_QMtest_opsPcheck_binary_ops_2() {
subroutine check_binary_ops_2()
  print *,  elem_func(i*iscalar)
! CHECK:  %[[VAL_30:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_38:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_39:.*]] = fir.load %[[VAL_38]] : !fir.ref<i32>
! CHECK:    %[[VAL_40:.*]] = arith.muli %[[VAL_39]], %{{.*}} : i32
! CHECK:    hlfir.yield_element %[[VAL_40]] : i32
! CHECK:  }
! CHECK:  %[[VAL_31:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_38:.*]] = hlfir.apply %[[VAL_30]], %[[ARG0]] : (!hlfir.expr<10xi32>, index) -> i32
! CHECK:    %[[VAL_39:.*]]:3 = hlfir.associate %[[VAL_38]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:    %[[VAL_40:.*]] = fir.call @_QPelem_func(%[[VAL_39]]#0) {{.*}} : (!fir.ref<i32>) -> i32
! CHECK:    hlfir.yield_element %[[VAL_40]] : i32
! CHECK:  }
end subroutine

! CHECK-LABEL: func.func @_QMtest_opsPcheck_negate() {
subroutine check_negate()
  print *,  elem_func(-i)
! CHECK:  %[[VAL_29:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_38:.*]] = fir.load %[[VAL_37]] : !fir.ref<i32>
! CHECK:    %[[VAL_39:.*]] = arith.subi %c0_i32, %[[VAL_38]] : i32
! CHECK:    hlfir.yield_element %[[VAL_39]] : i32
! CHECK:  }
! CHECK:  %[[VAL_30:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.apply %[[VAL_29]], %[[ARG0]] : (!hlfir.expr<10xi32>, index) -> i32
! CHECK:    %[[VAL_38:.*]]:3 = hlfir.associate %[[VAL_37]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:    %[[VAL_39:.*]] = fir.call @_QPelem_func(%[[VAL_38]]#0) {{.*}} : (!fir.ref<i32>) -> i32
! CHECK:    hlfir.yield_element %[[VAL_39]] : i32
! CHECK:  }
end subroutine

! CHECK-LABEL: func.func @_QMtest_opsPcheck_convert() {
subroutine check_convert()
  print *,  elem_func(int(x))
! CHECK:  %[[VAL_29:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10xf64>>, index) -> !fir.ref<f64>
! CHECK:    %[[VAL_38:.*]] = fir.load %[[VAL_37]] : !fir.ref<f64>
! CHECK:    %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (f64) -> i32
! CHECK:    hlfir.yield_element %[[VAL_39]] : i32
! CHECK:  }
! CHECK:  %[[VAL_30:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.apply %[[VAL_29]], %[[ARG0]] : (!hlfir.expr<10xi32>, index) -> i32
! CHECK:    %[[VAL_38:.*]]:3 = hlfir.associate %[[VAL_37]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:    %[[VAL_39:.*]] = fir.call @_QPelem_func(%[[VAL_38]]#0) {{.*}} : (!fir.ref<i32>) -> i32
! CHECK:    hlfir.yield_element %[[VAL_39]] : i32
! CHECK:  }
end subroutine

! CHECK-LABEL: func.func @_QMtest_opsPcheck_exteremum() {
subroutine check_exteremum()
  print *,  elem_func(min(i, j))
! CHECK:  %[[VAL_29:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_38:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_39:.*]] = fir.load %[[VAL_37]] : !fir.ref<i32>
! CHECK:    %[[VAL_40:.*]] = fir.load %[[VAL_38]] : !fir.ref<i32>
! CHECK:    %[[VAL_41:.*]] = arith.cmpi slt, %[[VAL_39]], %[[VAL_40]] : i32
! CHECK:    %[[VAL_42:.*]] = arith.select %[[VAL_41]], %[[VAL_39]], %[[VAL_40]] : i32
! CHECK:    hlfir.yield_element %[[VAL_42]] : i32
! CHECK:  }
! CHECK:  %[[VAL_30:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.apply %[[VAL_29]], %[[ARG0]] : (!hlfir.expr<10xi32>, index) -> i32
! CHECK:    %[[VAL_38:.*]]:3 = hlfir.associate %[[VAL_37]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:    %[[VAL_39:.*]] = fir.call @_QPelem_func(%[[VAL_38]]#0) {{.*}} : (!fir.ref<i32>) -> i32
! CHECK:    hlfir.yield_element %[[VAL_39]] : i32
! CHECK:  }
end subroutine

! CHECK-LABEL: func.func @_QMtest_opsPcheck_logical_unary_ops() {
subroutine check_logical_unary_ops()
  print *,  elem_func_logical(.not.b)
! CHECK:  %[[VAL_29:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<8>> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10x!fir.logical<8>>>, index) -> !fir.ref<!fir.logical<8>>
! CHECK:    %[[VAL_38:.*]] = fir.load %[[VAL_37]] : !fir.ref<!fir.logical<8>>
! CHECK:    %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (!fir.logical<8>) -> i1
! CHECK:    %[[VAL_40:.*]] = arith.xori %[[VAL_39]], %true : i1
! CHECK:    %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i1) -> !fir.logical<8>
! CHECK:    hlfir.yield_element %[[VAL_41]] : !fir.logical<8>
! CHECK:  }
! CHECK:  %[[VAL_30:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.apply %[[VAL_29]], %[[ARG0]] : (!hlfir.expr<10x!fir.logical<8>>, index) -> !fir.logical<8>
! CHECK:    %[[VAL_38:.*]]:3 = hlfir.associate %[[VAL_37]] {adapt.valuebyref} : (!fir.logical<8>) -> (!fir.ref<!fir.logical<8>>, !fir.ref<!fir.logical<8>>, i1)
! CHECK:    %[[VAL_39:.*]] = fir.call @_QPelem_func_logical(%[[VAL_38]]#0) {{.*}} : (!fir.ref<!fir.logical<8>>) -> i32
! CHECK:    hlfir.yield_element %[[VAL_39]] : i32
! CHECK:  }
end subroutine

! CHECK-LABEL: func.func @_QMtest_opsPcheck_logical_binary_ops() {
subroutine check_logical_binary_ops()
  print *,  elem_func_logical(a.eqv.b)
! CHECK:  %[[VAL_29:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<8>> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10x!fir.logical<8>>>, index) -> !fir.ref<!fir.logical<8>>
! CHECK:    %[[VAL_38:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10x!fir.logical<8>>>, index) -> !fir.ref<!fir.logical<8>>
! CHECK:    %[[VAL_39:.*]] = fir.load %[[VAL_37]] : !fir.ref<!fir.logical<8>>
! CHECK:    %[[VAL_40:.*]] = fir.load %[[VAL_38]] : !fir.ref<!fir.logical<8>>
! CHECK:    %[[VAL_41:.*]] = fir.convert %[[VAL_39]] : (!fir.logical<8>) -> i1
! CHECK:    %[[VAL_42:.*]] = fir.convert %[[VAL_40]] : (!fir.logical<8>) -> i1
! CHECK:    %[[VAL_43:.*]] = arith.cmpi eq, %[[VAL_41]], %[[VAL_42]] : i1
! CHECK:    %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i1) -> !fir.logical<8>
! CHECK:    hlfir.yield_element %[[VAL_44]] : !fir.logical<8>
! CHECK:  }
! CHECK:  %[[VAL_30:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.apply %[[VAL_29]], %[[ARG0]] : (!hlfir.expr<10x!fir.logical<8>>, index) -> !fir.logical<8>
! CHECK:    %[[VAL_38:.*]]:3 = hlfir.associate %[[VAL_37]] {adapt.valuebyref} : (!fir.logical<8>) -> (!fir.ref<!fir.logical<8>>, !fir.ref<!fir.logical<8>>, i1)
! CHECK:    %[[VAL_39:.*]] = fir.call @_QPelem_func_logical(%[[VAL_38]]#0) {{.*}} : (!fir.ref<!fir.logical<8>>) -> i32
! CHECK:    hlfir.yield_element %[[VAL_39]] : i32
! CHECK:  }
end subroutine

! CHECK-LABEL: func.func @_QMtest_opsPcheck_compare() {
subroutine check_compare()
  print *,  elem_func_logical4(x.lt.y)
! CHECK:  %[[VAL_29:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<4>> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10xf64>>, index) -> !fir.ref<f64>
! CHECK:    %[[VAL_38:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10xf64>>, index) -> !fir.ref<f64>
! CHECK:    %[[VAL_39:.*]] = fir.load %[[VAL_37]] : !fir.ref<f64>
! CHECK:    %[[VAL_40:.*]] = fir.load %[[VAL_38]] : !fir.ref<f64>
! CHECK:    %[[VAL_41:.*]] = arith.cmpf olt, %[[VAL_39]], %[[VAL_40]] {{.*}} : f64
! CHECK:    %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i1) -> !fir.logical<4>
! CHECK:    hlfir.yield_element %[[VAL_42]] : !fir.logical<4>
! CHECK:  }
! CHECK:  %[[VAL_30:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.apply %[[VAL_29]], %[[ARG0]] : (!hlfir.expr<10x!fir.logical<4>>, index) -> !fir.logical<4>
! CHECK:    %[[VAL_38:.*]]:3 = hlfir.associate %[[VAL_37]] {adapt.valuebyref} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:    %[[VAL_39:.*]] = fir.call @_QPelem_func_logical4(%[[VAL_38]]#0) {{.*}} : (!fir.ref<!fir.logical<4>>) -> i32
! CHECK:    hlfir.yield_element %[[VAL_39]] : i32
! CHECK:  }
end subroutine

! CHECK-LABEL: func.func @_QMtest_opsPcheck_pow() {
subroutine check_pow()
  print *,  elem_func_real(x**y)
! CHECK:  %[[VAL_29:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xf64> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10xf64>>, index) -> !fir.ref<f64>
! CHECK:    %[[VAL_38:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10xf64>>, index) -> !fir.ref<f64>
! CHECK:    %[[VAL_39:.*]] = fir.load %[[VAL_37]] : !fir.ref<f64>
! CHECK:    %[[VAL_40:.*]] = fir.load %[[VAL_38]] : !fir.ref<f64>
! CHECK:    %[[VAL_41:.*]] = math.powf %[[VAL_39]], %[[VAL_40]] {{.*}} : f64
! CHECK:    hlfir.yield_element %[[VAL_41]] : f64
! CHECK:  }
! CHECK:  %[[VAL_30:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.apply %[[VAL_29]], %[[ARG0]] : (!hlfir.expr<10xf64>, index) -> f64
! CHECK:    %[[VAL_38:.*]] = fir.call @_QPelem_func_real(%[[VAL_37]]) {{.*}} : (f64) -> i32
! CHECK:    hlfir.yield_element %[[VAL_38]] : i32
! CHECK:  }
end subroutine

! CHECK-LABEL: func.func @_QMtest_opsPcheck_cmplx_part() {
subroutine check_cmplx_part()
  print *,  elem_func_real(AIMAG(z1 + z2))
! CHECK:  %[[VAL_30:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xcomplex<f64>> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_39:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10xcomplex<f64>>>, index) -> !fir.ref<complex<f64>>
! CHECK:    %[[VAL_40:.*]] = fir.load %[[VAL_39]] : !fir.ref<complex<f64>>
! CHECK:    %[[VAL_41:.*]] = fir.addc %[[VAL_40]], %{{.*}} {{.*}} : complex<f64>
! CHECK:    hlfir.yield_element %[[VAL_41]] : complex<f64>
! CHECK:  }
! CHECK:  %[[VAL_31:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xf64> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_39:.*]] = hlfir.apply %[[VAL_30]], %[[ARG0]] : (!hlfir.expr<10xcomplex<f64>>, index) -> complex<f64>
! CHECK:    %[[VAL_40:.*]] = fir.extract_value %[[VAL_39]], [1 : index] : (complex<f64>) -> f64
! CHECK:    hlfir.yield_element %[[VAL_40]] : f64
! CHECK:  }
! CHECK:  %[[VAL_32:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_39:.*]] = hlfir.apply %[[VAL_31]], %[[ARG0]] : (!hlfir.expr<10xf64>, index) -> f64
! CHECK:    %[[VAL_40:.*]] = fir.call @_QPelem_func_real(%[[VAL_39]]) {{.*}} : (f64) -> i32
! CHECK:    hlfir.yield_element %[[VAL_40]] : i32
! CHECK:  }
end subroutine

! CHECK-LABEL: func.func @_QMtest_opsPcheck_parentheses() {
subroutine check_parentheses()
  print *,  elem_func_real((x))
! CHECK:  %[[VAL_29:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xf64> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10xf64>>, index) -> !fir.ref<f64>
! CHECK:    %[[VAL_38:.*]] = fir.load %[[VAL_37]] : !fir.ref<f64>
! CHECK:    %[[VAL_39:.*]] = hlfir.no_reassoc %[[VAL_38]] : f64
! CHECK:    hlfir.yield_element %[[VAL_39]] : f64
! CHECK:  }
! CHECK:  %[[VAL_30:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.apply %[[VAL_29]], %[[ARG0]] : (!hlfir.expr<10xf64>, index) -> f64
! CHECK:    %[[VAL_38:.*]] = fir.call @_QPelem_func_real(%[[VAL_37]]) {{.*}} : (f64) -> i32
! CHECK:    hlfir.yield_element %[[VAL_38]] : i32
! CHECK:  }
end subroutine

! CHECK-LABEL: func.func @_QMtest_opsPcheck_parentheses_logical() {
subroutine check_parentheses_logical()
  print *,  elem_func_logical((a))
! CHECK:  %[[VAL_29:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<8>> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.designate %{{.*}} (%[[ARG0]])  : (!fir.ref<!fir.array<10x!fir.logical<8>>>, index) -> !fir.ref<!fir.logical<8>>
! CHECK:    %[[VAL_38:.*]] = fir.load %[[VAL_37]] : !fir.ref<!fir.logical<8>>
! CHECK:    %[[VAL_39:.*]] = hlfir.no_reassoc %[[VAL_38]] : !fir.logical<8>
! CHECK:    hlfir.yield_element %[[VAL_39]] : !fir.logical<8>
! CHECK:  }
! CHECK:  %[[VAL_30:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:  ^bb0(%[[ARG0:.*]]: index):
! CHECK:    %[[VAL_37:.*]] = hlfir.apply %[[VAL_29]], %[[ARG0]] : (!hlfir.expr<10x!fir.logical<8>>, index) -> !fir.logical<8>
! CHECK:    %[[VAL_38:.*]]:3 = hlfir.associate %[[VAL_37]] {adapt.valuebyref} : (!fir.logical<8>) -> (!fir.ref<!fir.logical<8>>, !fir.ref<!fir.logical<8>>, i1)
! CHECK:    %[[VAL_39:.*]] = fir.call @_QPelem_func_logical(%[[VAL_38]]#0) {{.*}} : (!fir.ref<!fir.logical<8>>) -> i32
! CHECK:    hlfir.yield_element %[[VAL_39]] : i32
! CHECK:  }
end subroutine

subroutine check_parentheses_derived(a)
  type t
    integer :: i
  end type
  interface
    integer elemental function elem_func_derived(x)
      import :: t
      type(t), intent(in) :: x
    end function
  end interface
  type(t), pointer :: a(:)
  print *,  elem_func_derived((a))
! CHECK:  %[[VAL_42:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>> {
! CHECK:  ^bb0(%[[ARG1:.*]]: index):
! CHECK:    %[[VAL_53:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>>>, index) -> !fir.ref<!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>
! CHECK:    %[[VAL_54:.*]] = hlfir.as_expr %[[VAL_53]] : (!fir.ref<!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>) -> !hlfir.expr<!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>
! CHECK:    hlfir.yield_element %[[VAL_54]] : !hlfir.expr<!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>
! CHECK:  }
! CHECK:  %[[VAL_43:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:  ^bb0(%[[ARG1:.*]]: index):
! CHECK:    %[[VAL_50:.*]] = hlfir.apply %[[VAL_42]], %[[ARG1]] : (!hlfir.expr<?x!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>, index) -> !hlfir.expr<!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>
! CHECK:    %[[VAL_51:.*]]:3 = hlfir.associate %[[VAL_50]] {adapt.valuebyref} : (!hlfir.expr<!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>) -> (!fir.ref<!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>, !fir.ref<!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>, i1)
! CHECK:    %[[VAL_52:.*]] = fir.call @_QPelem_func_derived(%[[VAL_51]]#0) {{.*}} : (!fir.ref<!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>) -> i32
! CHECK:    hlfir.end_associate %[[VAL_51]]#1, %[[VAL_51]]#2 : !fir.ref<!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>, i1
! CHECK:    hlfir.yield_element %[[VAL_52]] : i32
! CHECK:  }
end subroutine
end module
