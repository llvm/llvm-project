! Test descriptor dummy argument preparation when the
! dummy has IGNORE_TKR(t). The descriptor should be prepared
! according to the actual argument type, but its bounds and
! attributes should still be set as expected for the dummy.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module tkr_ifaces
  interface
    subroutine takes_assumed_shape_ignore_tkr_t(x) bind(c)
      !dir$ ignore_tkr (t) x
      integer :: x(:)
    end subroutine
  end interface
end module

subroutine test_ignore_t_1(x)
  use tkr_ifaces
  real :: x(10)
  call takes_assumed_shape_ignore_tkr_t(x)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_ignore_t_1(
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_6:.*]] = fir.shift %[[VAL_5]] : (index) -> !fir.shift<1>
! CHECK:           %[[VAL_7:.*]] = fir.rebox %{{.*}}(%[[VAL_6]]) : (!fir.box<!fir.array<10xf32>>, !fir.shift<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @takes_assumed_shape_ignore_tkr_t(%[[VAL_8]]) proc_attrs<bind_c> fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()

subroutine test_ignore_t_2(x)
  use tkr_ifaces
  class(*) :: x(:)
  call takes_assumed_shape_ignore_tkr_t(x)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_ignore_t_2(
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_3:.*]] = fir.shift %[[VAL_2]] : (index) -> !fir.shift<1>
! CHECK:           %[[VAL_4:.*]] = fir.rebox %{{.*}}(%[[VAL_3]]) : (!fir.class<!fir.array<?xnone>>, !fir.shift<1>) -> !fir.class<!fir.array<?xnone>>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.class<!fir.array<?xnone>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @takes_assumed_shape_ignore_tkr_t(%[[VAL_5]]) proc_attrs<bind_c> fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()

subroutine test_ignore_t_3(x)
  use tkr_ifaces
  real :: x(10)
  call takes_assumed_shape_ignore_tkr_t(x+1.0)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_ignore_t_3(
! CHECK:           %[[VAL_12:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_13:.*]] = fir.shift %[[VAL_12]] : (index) -> !fir.shift<1>
! CHECK:           %[[VAL_14:.*]] = fir.rebox %{{.*}}(%[[VAL_13]]) : (!fir.box<!fir.array<10xf32>>, !fir.shift<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @takes_assumed_shape_ignore_tkr_t(%[[VAL_15]]) proc_attrs<bind_c> fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()

subroutine test_ignore_t_4(x)
  use tkr_ifaces
  real, pointer :: x(:)
  call takes_assumed_shape_ignore_tkr_t(x)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_ignore_t_4(
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_4:.*]] = fir.shift %[[VAL_3]] : (index) -> !fir.shift<1>
! CHECK:           %[[VAL_5:.*]] = fir.rebox %{{.*}}(%[[VAL_4]]) : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @takes_assumed_shape_ignore_tkr_t(%[[VAL_6]]) proc_attrs<bind_c> fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
