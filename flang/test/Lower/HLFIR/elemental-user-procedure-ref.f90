! Test lowering of user defined elemental procedure reference to HLFIR
! RUN: bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

subroutine by_addr(x, y)
  integer :: x
  real :: y(100)
  interface
    real elemental function elem(a, b)
      integer, intent(in) :: a
      real, intent(in) :: b
    end function
  end interface
  call baz(elem(x, y))
end subroutine
! CHECK-LABEL: func.func @_QPby_addr(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0:.*]] {{.*}}x
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1:.*]](%[[VAL_4:[^)]*]]) {{.*}}y
! CHECK:  %[[VAL_6:.*]] = hlfir.elemental %[[VAL_4]] : (!fir.shape<1>) -> !hlfir.expr<100xf32> {
! CHECK:  ^bb0(%[[VAL_7:.*]]: index):
! CHECK:    %[[VAL_8:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_7]])  : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:    %[[VAL_9:.*]] = fir.call @_QPelem(%[[VAL_2]]#1, %[[VAL_8]]) fastmath<contract> : (!fir.ref<i32>, !fir.ref<f32>) -> f32
! CHECK:    hlfir.yield_element %[[VAL_9]] : f32
! CHECK:  }
! CHECK: fir.call
! CHECK: hlfir.destroy %[[VAL_6]]

subroutine by_value(x, y)
  integer :: x
  real :: y(10, 20)
  interface
    real elemental function elem_val(a, b)
      integer, value :: a
      real, value :: b
    end function
  end interface
  call baz(elem_val(x, y))
end subroutine
! CHECK-LABEL: func.func @_QPby_value(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0:.*]] {{.*}}x
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1:.*]](%[[VAL_5:[^)]*]]) {{.*}}y
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_8:.*]] = hlfir.elemental %[[VAL_5]] : (!fir.shape<2>) -> !hlfir.expr<10x20xf32> {
! CHECK:  ^bb0(%[[VAL_9:.*]]: index, %[[VAL_10:.*]]: index):
! CHECK:    %[[VAL_11:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_9]], %[[VAL_10]])  : (!fir.ref<!fir.array<10x20xf32>>, index, index) -> !fir.ref<f32>
! CHECK:    %[[VAL_12:.*]] = fir.load %[[VAL_11]] : !fir.ref<f32>
! CHECK:    %[[VAL_13:.*]] = fir.call @_QPelem_val(%[[VAL_7]], %[[VAL_12]]) fastmath<contract> : (i32, f32) -> f32
! CHECK:    hlfir.yield_element %[[VAL_13]] : f32
! CHECK:  }
! CHECK: fir.call
! CHECK: hlfir.destroy %[[VAL_8]]

subroutine by_boxaddr(x, y)
  character(*) :: x
  character(*) :: y(100)
  interface
    real elemental function char_elem(a, b)
      character(*), intent(in) :: a
      character(*), intent(in) :: b
    end function
  end interface
  call baz2(char_elem(x, y))
end subroutine
! CHECK-LABEL: func.func @_QPby_boxaddr(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2:.*]]#0 typeparams %[[VAL_2]]#1 {{.*}}x
! CHECK:  %[[VAL_6:.*]] = arith.constant 100 : index
! CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_5:.*]](%[[VAL_7:.*]]) typeparams %[[VAL_4:.*]]#1 {{.*}}y
! CHECK:  %[[VAL_9:.*]] = hlfir.elemental %[[VAL_7]] : (!fir.shape<1>) -> !hlfir.expr<100xf32> {
! CHECK:  ^bb0(%[[VAL_10:.*]]: index):
! CHECK:    %[[VAL_11:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_10]])  typeparams %[[VAL_4]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:    %[[VAL_12:.*]] = fir.call @_QPchar_elem(%[[VAL_3]]#0, %[[VAL_11]]) fastmath<contract> : (!fir.boxchar<1>, !fir.boxchar<1>) -> f32
! CHECK:    hlfir.yield_element %[[VAL_12]] : f32
! CHECK:  }
! CHECK: fir.call
! CHECK: hlfir.destroy %[[VAL_9]]

subroutine sub(x, y)
  integer :: x
  real :: y(10, 20)
  interface
    elemental subroutine elem_sub(a, b)
      integer, intent(in) :: a
      real, intent(in) :: b
    end subroutine
  end interface
  call elem_sub(x, y)
end subroutine
! CHECK-LABEL: func.func @_QPsub(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0:.*]] {{.*}}x
! CHECK:  %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_4:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1:.*]](%[[VAL_5:[^)]*]]) {{.*}}y
! CHECK:  %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:  fir.do_loop %[[VAL_8:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_7]] {
! CHECK:    fir.do_loop %[[VAL_9:.*]] = %[[VAL_7]] to %[[VAL_3]] step %[[VAL_7]] {
! CHECK:      %[[VAL_10:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_9]], %[[VAL_8]])  : (!fir.ref<!fir.array<10x20xf32>>, index, index) -> !fir.ref<f32>
! CHECK:      fir.call @_QPelem_sub(%[[VAL_2]]#1, %[[VAL_10]]) fastmath<contract> : (!fir.ref<i32>, !fir.ref<f32>) -> ()
! CHECK:    }
! CHECK:  }
