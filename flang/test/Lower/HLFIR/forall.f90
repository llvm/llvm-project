! Test lowering of Forall to HLFIR.
! RUN: bbc --hlfir -o - %s | FileCheck %s

module forall_defs
  integer :: x(10, 10), y(10)
  interface
    pure integer(8) function ifoo2(i, j)
      integer(8), value :: i, j
    end function
    pure integer(8) function jfoo()
    end function
    pure integer(8) function jbar()
    end function
    pure logical function predicate(i)
      integer(8), intent(in) :: i
    end function
  end interface
end module

subroutine test_simple_forall()
  use forall_defs
  forall (integer(8)::i=1:10) x(i, i) = y(i)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_simple_forall() {
! CHECK:  %[[VAL_0:.*]] = arith.constant 10 : i32
! CHECK:  %[[VAL_1:.*]] = arith.constant 1 : i32
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare {{.*}}Ey
! CHECK:  hlfir.forall lb {
! CHECK:    hlfir.yield %[[VAL_1]] : i32
! CHECK:  } ub {
! CHECK:    hlfir.yield %[[VAL_0]] : i32
! CHECK:  }  (%[[VAL_9:.*]]: i64) {
! CHECK:    hlfir.region_assign {
! CHECK:      %[[VAL_10:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_9]])  : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK:      %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
! CHECK:      hlfir.yield %[[VAL_11]] : i32
! CHECK:    } to {
! CHECK:      %[[VAL_12:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_9]], %[[VAL_9]])  : (!fir.ref<!fir.array<10x10xi32>>, i64, i64) -> !fir.ref<i32>
! CHECK:      hlfir.yield %[[VAL_12]] : !fir.ref<i32>
! CHECK:    }
! CHECK:  }

subroutine test_forall_step(step)
  use forall_defs
  integer :: step
  forall (integer(8)::i=1:10:step) x(i, i) = y(i)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_forall_step(
! CHECK:  %[[VAL_1:.*]] = arith.constant 10 : i32
! CHECK:  %[[VAL_2:.*]] = arith.constant 1 : i32
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare {{.*}}Estep
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  %[[VAL_10:.*]]:2 = hlfir.declare {{.*}}Ey
! CHECK:  %[[VAL_11:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:  hlfir.forall lb {
! CHECK:    hlfir.yield %[[VAL_2]] : i32
! CHECK:  } ub {
! CHECK:    hlfir.yield %[[VAL_1]] : i32
! CHECK:  } step {
! CHECK:    hlfir.yield %[[VAL_11]] : i32
! CHECK:  }  (%[[VAL_12:.*]]: i64) {
! CHECK:    hlfir.region_assign {
! CHECK:      %[[VAL_13:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_12]])  : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK:      %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
! CHECK:      hlfir.yield %[[VAL_14]] : i32
! CHECK:    } to {
! CHECK:      %[[VAL_15:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_12]], %[[VAL_12]])  : (!fir.ref<!fir.array<10x10xi32>>, i64, i64) -> !fir.ref<i32>
! CHECK:      hlfir.yield %[[VAL_15]] : !fir.ref<i32>
! CHECK:    }
! CHECK:  }

subroutine test_forall_mask()
  use forall_defs
  forall (integer(8)::i=1:10, predicate(i)) x(i, i) = y(i)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_forall_mask() {
! CHECK:  %[[VAL_0:.*]] = arith.constant 10 : i32
! CHECK:  %[[VAL_1:.*]] = arith.constant 1 : i32
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare {{.*}}Ey
! CHECK:  hlfir.forall lb {
! CHECK:    hlfir.yield %[[VAL_1]] : i32
! CHECK:  } ub {
! CHECK:    hlfir.yield %[[VAL_0]] : i32
! CHECK:  }  (%[[VAL_9:.*]]: i64) {
! CHECK:    %[[VAL_10:.*]] = hlfir.forall_index "i" %[[VAL_9]] : (i64) -> !fir.ref<i64>
! CHECK:    hlfir.forall_mask {
! CHECK:      %[[VAL_11:.*]] = fir.call @_QPpredicate(%[[VAL_10]]) proc_attrs<pure> fastmath<contract> : (!fir.ref<i64>) -> !fir.logical<4>
! CHECK:      %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.logical<4>) -> i1
! CHECK:      hlfir.yield %[[VAL_12]] : i1
! CHECK:    } do {
! CHECK:      hlfir.region_assign {
! CHECK:        %[[I_LOAD:.*]] = fir.load %[[VAL_10]] : !fir.ref<i64>
! CHECK:        %[[VAL_13:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[I_LOAD]])  : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK:        %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
! CHECK:        hlfir.yield %[[VAL_14]] : i32
! CHECK:      } to {
! CHECK:        %[[I_LOAD:.*]] = fir.load %[[VAL_10]] : !fir.ref<i64>
! CHECK:        %[[VAL_15:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[I_LOAD]], %[[I_LOAD]])  : (!fir.ref<!fir.array<10x10xi32>>, i64, i64) -> !fir.ref<i32>
! CHECK:        hlfir.yield %[[VAL_15]] : !fir.ref<i32>
! CHECK:      }
! CHECK:    }
! CHECK:  }

subroutine test_forall_several_indices()
  use forall_defs
  ! Test outer forall controls are lowered outside.
  forall (integer(8)::i=ibar():ifoo(), j=jfoo():jbar()) x(i, j) = y(ifoo2(i, j))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_forall_several_indices() {
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare {{.*}}Ey
! CHECK:  %[[VAL_7:.*]] = fir.call @_QPibar() fastmath<contract> : () -> i32
! CHECK:  %[[VAL_8:.*]] = fir.call @_QPifoo() fastmath<contract> : () -> i32
! CHECK:  %[[VAL_9:.*]] = fir.call @_QPjfoo() proc_attrs<pure> fastmath<contract> : () -> i64
! CHECK:  %[[VAL_10:.*]] = fir.call @_QPjbar() proc_attrs<pure> fastmath<contract> : () -> i64
! CHECK:  hlfir.forall lb {
! CHECK:    hlfir.yield %[[VAL_7]] : i32
! CHECK:  } ub {
! CHECK:    hlfir.yield %[[VAL_8]] : i32
! CHECK:  }  (%[[VAL_11:.*]]: i64) {
! CHECK:    hlfir.forall lb {
! CHECK:      hlfir.yield %[[VAL_9]] : i64
! CHECK:    } ub {
! CHECK:      hlfir.yield %[[VAL_10]] : i64
! CHECK:    }  (%[[VAL_12:.*]]: i64) {
! CHECK:      hlfir.region_assign {
! CHECK:        %[[VAL_13:.*]] = fir.call @_QPifoo2(%[[VAL_11]], %[[VAL_12]]) proc_attrs<pure> fastmath<contract> : (i64, i64) -> i64
! CHECK:        %[[VAL_14:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK:        %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
! CHECK:        hlfir.yield %[[VAL_15]] : i32
! CHECK:      } to {
! CHECK:        %[[VAL_16:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_11]], %[[VAL_12]])  : (!fir.ref<!fir.array<10x10xi32>>, i64, i64) -> !fir.ref<i32>
! CHECK:        hlfir.yield %[[VAL_16]] : !fir.ref<i32>
! CHECK:      }
! CHECK:    }
! CHECK:  }

subroutine test_nested_foralls()
  use forall_defs
  forall (integer(8)::i=1:10)
    x(i, i) = y(i)
    ! ifoo and ibar could depend on x since it is a module
    ! variable use associated. The calls in the control value
    ! computation cannot be hoisted from the outer forall
    ! even when they do not depend on outer forall indices.
    forall (integer(8)::j=jfoo():jbar())
      x(i, j) = x(j, i)
    end forall
  end forall
end subroutine
! CHECK-LABEL:   func.func @_QPtest_nested_foralls() {
! CHECK:  %[[VAL_0:.*]] = arith.constant 10 : i32
! CHECK:  %[[VAL_1:.*]] = arith.constant 1 : i32
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare {{.*}}Ey
! CHECK:  hlfir.forall lb {
! CHECK:    hlfir.yield %[[VAL_1]] : i32
! CHECK:  } ub {
! CHECK:    hlfir.yield %[[VAL_0]] : i32
! CHECK:  }  (%[[VAL_9:.*]]: i64) {
! CHECK:    hlfir.region_assign {
! CHECK:      %[[VAL_10:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_9]])  : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK:      %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
! CHECK:      hlfir.yield %[[VAL_11]] : i32
! CHECK:    } to {
! CHECK:      %[[VAL_12:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_9]], %[[VAL_9]])  : (!fir.ref<!fir.array<10x10xi32>>, i64, i64) -> !fir.ref<i32>
! CHECK:      hlfir.yield %[[VAL_12]] : !fir.ref<i32>
! CHECK:    }
! CHECK:    hlfir.forall lb {
! CHECK:      %[[VAL_13:.*]] = fir.call @_QPjfoo() proc_attrs<pure> fastmath<contract> : () -> i64
! CHECK:      hlfir.yield %[[VAL_13]] : i64
! CHECK:    } ub {
! CHECK:      %[[VAL_14:.*]] = fir.call @_QPjbar() proc_attrs<pure> fastmath<contract> : () -> i64
! CHECK:      hlfir.yield %[[VAL_14]] : i64
! CHECK:    }  (%[[VAL_15:.*]]: i64) {
! CHECK:      hlfir.region_assign {
! CHECK:        %[[VAL_16:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_15]], %[[VAL_9]])  : (!fir.ref<!fir.array<10x10xi32>>, i64, i64) -> !fir.ref<i32>
! CHECK:        %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<i32>
! CHECK:        hlfir.yield %[[VAL_17]] : i32
! CHECK:      } to {
! CHECK:        %[[VAL_18:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_9]], %[[VAL_15]])  : (!fir.ref<!fir.array<10x10xi32>>, i64, i64) -> !fir.ref<i32>
! CHECK:        hlfir.yield %[[VAL_18]] : !fir.ref<i32>
! CHECK:      }
! CHECK:    }
! CHECK:  }
