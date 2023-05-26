! Test lowering of WHERE construct and statements to HLFIR.
! RUN: bbc --hlfir -emit-fir -o - %s | FileCheck %s

module where_defs
  logical :: mask(10)
  real :: x(10), y(10)
  real, allocatable :: a(:), b(:)
  interface
    function return_temporary_mask()
      logical, allocatable :: return_temporary_mask(:)
    end function
    function return_temporary_array()
      real, allocatable :: return_temporary_array(:)
    end function
  end interface
end module

subroutine simple_where()
  use where_defs, only: mask, x, y
  where (mask) x = y
end subroutine
! CHECK-LABEL:   func.func @_QPsimple_where() {
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}Emask
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  %[[VAL_11:.*]]:2 = hlfir.declare {{.*}}Ey
! CHECK:  hlfir.where {
! CHECK:    hlfir.yield %[[VAL_3]]#0 : !fir.ref<!fir.array<10x!fir.logical<4>>>
! CHECK:  } do {
! CHECK:    hlfir.region_assign {
! CHECK:      hlfir.yield %[[VAL_11]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK:    } to {
! CHECK:      hlfir.yield %[[VAL_7]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK:    }
! CHECK:  }
! CHECK:  return
! CHECK:}

subroutine where_construct()
  use where_defs
  where (mask)
    x = y
    a = b
  end where
end subroutine
! CHECK-LABEL:   func.func @_QPwhere_construct() {
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMwhere_defsEa"}
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMwhere_defsEb"}
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare {{.*}}Emask
! CHECK:  %[[VAL_11:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  %[[VAL_15:.*]]:2 = hlfir.declare {{.*}}Ey
! CHECK:  hlfir.where {
! CHECK:    hlfir.yield %[[VAL_7]]#0 : !fir.ref<!fir.array<10x!fir.logical<4>>>
! CHECK:  } do {
! CHECK:    hlfir.region_assign {
! CHECK:      hlfir.yield %[[VAL_15]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK:    } to {
! CHECK:      hlfir.yield %[[VAL_11]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK:    }
! CHECK:    hlfir.region_assign {
! CHECK:      %[[VAL_16:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:      hlfir.yield %[[VAL_16]] : !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:    } to {
! CHECK:      %[[VAL_17:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:      hlfir.yield %[[VAL_17]] : !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:    }
! CHECK:  }
! CHECK:  return
! CHECK:}

subroutine where_cleanup()
  use where_defs, only: x, return_temporary_mask, return_temporary_array
  where (return_temporary_mask()) x = return_temporary_array()
end subroutine
! CHECK-LABEL:   func.func @_QPwhere_cleanup() {
! CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = ".result"}
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>> {bindc_name = ".result"}
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  hlfir.where {
! CHECK:    %[[VAL_6:.*]] = fir.call @_QPreturn_temporary_mask() fastmath<contract> : () -> !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
! CHECK:    fir.save_result %[[VAL_6]] to %[[VAL_1]] : !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK:    %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>)
! CHECK:    %[[VAL_8:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK:    hlfir.yield %[[VAL_8]] : !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>> cleanup {
! CHECK:        fir.freemem
! CHECK:    }
! CHECK:  } do {
! CHECK:    hlfir.region_assign {
! CHECK:      %[[VAL_14:.*]] = fir.call @_QPreturn_temporary_array() fastmath<contract> : () -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:      fir.save_result %[[VAL_14]] to %[[VAL_0]] : !fir.box<!fir.heap<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:      %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
! CHECK:      %[[VAL_16:.*]] = fir.load %[[VAL_15]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:      hlfir.yield %[[VAL_16]] : !fir.box<!fir.heap<!fir.array<?xf32>>> cleanup {
! CHECK:          fir.freemem
! CHECK:      }
! CHECK:    } to {
! CHECK:      hlfir.yield %[[VAL_5]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK:    }
! CHECK:  }

subroutine simple_elsewhere()
  use where_defs
  where (mask)
    x = y
  elsewhere
    y = x
  end where
end subroutine
! CHECK-LABEL:   func.func @_QPsimple_elsewhere() {
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare {{.*}}Emask
! CHECK:  %[[VAL_11:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  %[[VAL_15:.*]]:2 = hlfir.declare {{.*}}Ey
! CHECK:  hlfir.where {
! CHECK:    hlfir.yield %[[VAL_7]]#0 : !fir.ref<!fir.array<10x!fir.logical<4>>>
! CHECK:  } do {
! CHECK:    hlfir.region_assign {
! CHECK:      hlfir.yield %[[VAL_15]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK:    } to {
! CHECK:      hlfir.yield %[[VAL_11]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK:    }
! CHECK:    hlfir.elsewhere do {
! CHECK:      hlfir.region_assign {
! CHECK:        hlfir.yield %[[VAL_11]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK:      } to {
! CHECK:        hlfir.yield %[[VAL_15]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK:      }
! CHECK:    }
! CHECK:  }

subroutine elsewhere_2(mask2)
  use where_defs, only : mask, x, y
  logical :: mask2(:)
  where (mask)
    x = y
  elsewhere(mask2)
    y = x
  elsewhere
    x = foo()
  end where
end subroutine
! CHECK-LABEL:   func.func @_QPelsewhere_2(
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}Emask
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare {{.*}}Emask2
! CHECK:  %[[VAL_11:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  %[[VAL_15:.*]]:2 = hlfir.declare {{.*}}Ey
! CHECK:  hlfir.where {
! CHECK:    hlfir.yield %[[VAL_5]]#0 : !fir.ref<!fir.array<10x!fir.logical<4>>>
! CHECK:  } do {
! CHECK:    hlfir.region_assign {
! CHECK:      hlfir.yield %[[VAL_15]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK:    } to {
! CHECK:      hlfir.yield %[[VAL_11]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK:    }
! CHECK:    hlfir.elsewhere mask {
! CHECK:      hlfir.yield %[[VAL_6]]#0 : !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:    } do {
! CHECK:      hlfir.region_assign {
! CHECK:        hlfir.yield %[[VAL_11]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK:      } to {
! CHECK:        hlfir.yield %[[VAL_15]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK:      }
! CHECK:      hlfir.elsewhere do {
! CHECK:        hlfir.region_assign {
! CHECK:          %[[VAL_16:.*]] = fir.call @_QPfoo() fastmath<contract> : () -> f32
! CHECK:          hlfir.yield %[[VAL_16]] : f32
! CHECK:        } to {
! CHECK:          hlfir.yield %[[VAL_11]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK:        }
! CHECK:      }
! CHECK:    }
! CHECK:  }
