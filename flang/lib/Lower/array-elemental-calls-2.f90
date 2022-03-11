! RUN: bbc -o - -emit-fir %s | FileCheck %s

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
  ! CHECK-LABEL: func @_QMtest_opsPcheck_binary_ops() {
  subroutine check_binary_ops()
    print *,  elem_func(i+j)
  ! CHECK:  %[[VAL_0:.*]] = fir.alloca i32
  ! CHECK:  fir.do_loop
  ! CHECK:  %[[VAL_25:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10xi32>, index) -> i32
  ! CHECK:  %[[VAL_26:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10xi32>, index) -> i32
  ! CHECK:  %[[VAL_27:.*]] = arith.addi %[[VAL_25]], %[[VAL_26]] : i32
  ! CHECK:  fir.store %[[VAL_27]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:  fir.call @_QPelem_func(%[[VAL_0]]) : (!fir.ref<i32>) -> i32
  end subroutine
  
  ! CHECK-LABEL: func @_QMtest_opsPcheck_binary_ops_2() {
  subroutine check_binary_ops_2()
    print *,  elem_func(i*iscalar)
  ! CHECK:  %[[VAL_0:.*]] = fir.alloca i32
  ! CHECK:  %[[VAL_13:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK:  fir.do_loop
  ! CHECK:  %[[VAL_25:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10xi32>, index) -> i32
  ! CHECK:  %[[VAL_27:.*]] = arith.muli %[[VAL_25]], %[[VAL_13]] : i32
  ! CHECK:  fir.store %[[VAL_27]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:  fir.call @_QPelem_func(%[[VAL_0]]) : (!fir.ref<i32>) -> i32
  end subroutine
  
  ! CHECK-LABEL: func @_QMtest_opsPcheck_negate() {
  subroutine check_negate()
    print *,  elem_func(-i)
  ! CHECK:  %[[VAL_0:.*]] = fir.alloca i32
  ! CHECK:  fir.do_loop
  ! CHECK:  %[[VAL_21:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10xi32>, index) -> i32
  ! CHECK:  %[[VAL_22:.*]] = arith.constant 0 : i32
  ! CHECK:  %[[VAL_23:.*]] = arith.subi %[[VAL_22]], %[[VAL_21]] : i32
  ! CHECK:  fir.store %[[VAL_23]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:  fir.call @_QPelem_func(%[[VAL_0]]) : (!fir.ref<i32>) -> i32
  end subroutine
  
  ! CHECK-LABEL: func @_QMtest_opsPcheck_convert() {
  subroutine check_convert()
    print *,  elem_func(int(x))
  ! CHECK:  %[[VAL_0:.*]] = fir.alloca i32
  ! CHECK:  fir.do_loop
  ! CHECK:  %[[VAL_21:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10xf64>, index) -> f64
  ! CHECK:  %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (f64) -> i32
  ! CHECK:  fir.store %[[VAL_22]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:  fir.call @_QPelem_func(%[[VAL_0]]) : (!fir.ref<i32>) -> i32
  end subroutine
  
  ! CHECK-LABEL: func @_QMtest_opsPcheck_exteremum() {
  subroutine check_exteremum()
    print *,  elem_func(min(i, j))
  ! CHECK:  %[[VAL_0:.*]] = fir.alloca i32
  ! CHECK:  fir.do_loop
  ! CHECK:  %[[VAL_25:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10xi32>, index) -> i32
  ! CHECK:  %[[VAL_26:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10xi32>, index) -> i32
  ! CHECK:  %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_25]], %[[VAL_26]] : i32
  ! CHECK:  %[[VAL_28:.*]] = select %[[VAL_27]], %[[VAL_25]], %[[VAL_26]] : i32
  ! CHECK:  fir.store %[[VAL_28]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:  fir.call @_QPelem_func(%[[VAL_0]]) : (!fir.ref<i32>) -> i32
  end subroutine
  
  ! CHECK-LABEL: func @_QMtest_opsPcheck_logical_unary_ops() {
  subroutine check_logical_unary_ops()
    print *,  elem_func_logical(.not.b)
  ! CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.logical<8>
  ! CHECK:  %[[VAL_12:.*]] = arith.constant true
  ! CHECK:  fir.do_loop
  ! CHECK:  %[[VAL_22:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10x!fir.logical<8>>, index) -> !fir.logical<8>
  ! CHECK:  %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (!fir.logical<8>) -> i1
  ! CHECK:  %[[VAL_24:.*]] = arith.xori %[[VAL_23]], %[[VAL_12]] : i1
  ! CHECK:  %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i1) -> !fir.logical<8>
  ! CHECK:  fir.store %[[VAL_25]] to %[[VAL_0]] : !fir.ref<!fir.logical<8>>
  ! CHECK:  fir.call @_QPelem_func_logical(%[[VAL_0]]) : (!fir.ref<!fir.logical<8>>) -> i32
  end subroutine
  
  ! CHECK-LABEL: func @_QMtest_opsPcheck_logical_binary_ops() {
  subroutine check_logical_binary_ops()
    print *,  elem_func_logical(a.eqv.b)
  ! CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.logical<8>
  ! CHECK:  fir.do_loop
  ! CHECK:  %[[VAL_25:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10x!fir.logical<8>>, index) -> !fir.logical<8>
  ! CHECK:  %[[VAL_26:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10x!fir.logical<8>>, index) -> !fir.logical<8>
  ! CHECK:  %[[VAL_27:.*]] = fir.convert %[[VAL_25]] : (!fir.logical<8>) -> i1
  ! CHECK:  %[[VAL_28:.*]] = fir.convert %[[VAL_26]] : (!fir.logical<8>) -> i1
  ! CHECK:  %[[VAL_29:.*]] = arith.cmpi eq, %[[VAL_27]], %[[VAL_28]] : i1
  ! CHECK:  %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i1) -> !fir.logical<8>
  ! CHECK:  fir.store %[[VAL_30]] to %[[VAL_0]] : !fir.ref<!fir.logical<8>>
  ! CHECK:  fir.call @_QPelem_func_logical(%[[VAL_0]]) : (!fir.ref<!fir.logical<8>>) -> i32
  end subroutine
  
  ! CHECK-LABEL: func @_QMtest_opsPcheck_compare() {
  subroutine check_compare()
    print *,  elem_func_logical4(x.lt.y)
  ! CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.logical<4>
  ! CHECK:  fir.do_loop
  ! CHECK:  %[[VAL_25:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10xf64>, index) -> f64
  ! CHECK:  %[[VAL_26:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10xf64>, index) -> f64
  ! CHECK:  %[[VAL_27:.*]] = arith.cmpf olt, %[[VAL_25]], %[[VAL_26]] : f64
  ! CHECK:  %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i1) -> !fir.logical<4>
  ! CHECK:  fir.store %[[VAL_28]] to %[[VAL_0]] : !fir.ref<!fir.logical<4>>
  ! CHECK:  fir.call @_QPelem_func_logical4(%[[VAL_0]]) : (!fir.ref<!fir.logical<4>>) -> i32
  end subroutine
  
  ! CHECK-LABEL: func @_QMtest_opsPcheck_pow() {
  subroutine check_pow()
    print *,  elem_func_real(x**y)
  ! CHECK:  %[[VAL_0:.*]] = fir.alloca f64
  ! CHECK:  fir.do_loop
  ! CHECK:  %[[VAL_25:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10xf64>, index) -> f64
  ! CHECK:  %[[VAL_26:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10xf64>, index) -> f64
  ! CHECK:  %[[VAL_27:.*]] = fir.call @__fd_pow_1(%[[VAL_25]], %[[VAL_26]]) : (f64, f64) -> f64
  ! CHECK:  fir.store %[[VAL_27]] to %[[VAL_0]] : !fir.ref<f64>
  ! CHECK:  %[[VAL_28:.*]] = fir.call @_QPelem_func_real(%[[VAL_0]]) : (!fir.ref<f64>) -> i32
  end subroutine
  
  ! CHECK-LABEL: func @_QMtest_opsPcheck_cmplx_part() {
  subroutine check_cmplx_part()
    print *,  elem_func_real(AIMAG(z1 + z2))
  ! CHECK:  %[[VAL_0:.*]] = fir.alloca f64
  ! CHECK:  %[[VAL_13:.*]] = fir.load %{{.*}} : !fir.ref<!fir.complex<8>>
  ! CHECK:  fir.do_loop
  ! CHECK:  %[[VAL_23:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10x!fir.complex<8>>, index) -> !fir.complex<8>
  ! CHECK:  %[[VAL_24:.*]] = fir.addc %[[VAL_23]], %[[VAL_13]] : !fir.complex<8>
  ! CHECK:  %[[VAL_25:.*]] = fir.extract_value %[[VAL_24]], [1 : index] : (!fir.complex<8>) -> f64
  ! CHECK:  fir.store %[[VAL_25]] to %[[VAL_0]] : !fir.ref<f64>
  ! CHECK:  fir.call @_QPelem_func_real(%[[VAL_0]]) : (!fir.ref<f64>) -> i32
  end subroutine
  
  ! CHECK-LABEL: func @_QMtest_opsPcheck_parentheses() {
  subroutine check_parentheses()
    print *,  elem_func_real((x))
  ! CHECK:  %[[VAL_0:.*]] = fir.alloca f64
  ! CHECK:  fir.do_loop
  ! CHECK:  %[[VAL_21:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10xf64>, index) -> f64
  ! CHECK:  %[[VAL_22:.*]] = fir.no_reassoc %[[VAL_21]] : f64
  ! CHECK:  fir.store %[[VAL_22]] to %[[VAL_0]] : !fir.ref<f64>
  ! CHECK:  fir.call @_QPelem_func_real(%[[VAL_0]]) : (!fir.ref<f64>) -> i32
  end subroutine
  
  ! CHECK-LABEL: func @_QMtest_opsPcheck_parentheses_logical() {
  subroutine check_parentheses_logical()
    print *,  elem_func_logical((a))
  ! CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.logical<8>
  ! CHECK:  fir.do_loop
  ! CHECK:  %[[VAL_21:.*]] = fir.array_fetch %{{.*}}, %{{.*}} : (!fir.array<10x!fir.logical<8>>, index) -> !fir.logical<8>
  ! CHECK:  %[[VAL_22:.*]] = fir.no_reassoc %[[VAL_21]] : !fir.logical<8>
  ! CHECK:  fir.store %[[VAL_22]] to %[[VAL_0]] : !fir.ref<!fir.logical<8>>
  ! CHECK:  fir.call @_QPelem_func_logical(%[[VAL_0]]) : (!fir.ref<!fir.logical<8>>) -> i32
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
  ! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>
  ! CHECK: fir.do_loop
  ! CHECK: %[[VAL_21:.*]] = fir.array_access %{{.}}, %{{.*}}
  ! CHECK: %[[VAL_22:.*]] = fir.no_reassoc %[[VAL_21]] : !fir.ref<!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>
  ! CHECK: %[[FIELD:.*]] = fir.field_index i, !fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>
  ! CHECK: %[[FROM:.*]] = fir.coordinate_of %[[VAL_22]], %[[FIELD]] : (!fir.ref<!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK: %[[TO:.*]] = fir.coordinate_of %[[VAL_0]], %[[FIELD]] : (!fir.ref<!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK: %[[VAL:.*]] = fir.load %[[FROM]] : !fir.ref<i32>
  ! CHECK: fir.store %[[VAL]] to %[[TO]] : !fir.ref<i32>
  ! CHECK: %25 = fir.call @_QPelem_func_derived(%[[VAL_0]]) : (!fir.ref<!fir.type<_QMtest_opsFcheck_parentheses_derivedTt{i:i32}>>) -> i32
  end subroutine
  end module
  