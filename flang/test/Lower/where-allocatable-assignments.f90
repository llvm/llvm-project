! Test that WHERE constructs containing assignments to whole allocatables
! lower to the expected HLFIR shape.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module mtest
contains

! CHECK-LABEL: func.func @_QMmtestPfoo(
! CHECK-SAME:       %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:       %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "b"}) {
subroutine foo(a, b)
  integer :: a(:)
  integer, allocatable :: b(:)
! CHECK:  %[[A:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} {uniq_name = "_QMmtestFfooEa"}
! CHECK:  %[[B:.*]]:2 = hlfir.declare %[[VAL_1]] {{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmtestFfooEb"}
        ! WHERE construct: mask, region_assign, elsewhere
! CHECK:  hlfir.where {
! CHECK:    %[[BOX_B:.*]] = fir.load %[[B]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:    %[[ELEM:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
! CHECK:      arith.cmpi sgt, %{{.*}}, %{{.*}} : i32
! CHECK:    }
! CHECK:    hlfir.yield %[[ELEM]] : !hlfir.expr<?x!fir.logical<4>> cleanup {
! CHECK:      hlfir.destroy %[[ELEM]] : !hlfir.expr<?x!fir.logical<4>>
! CHECK:    }
! CHECK:  } do {
          ! First assignment to a whole allocatable (in WHERE): b = a
! CHECK:    hlfir.region_assign {
! CHECK:      hlfir.yield %[[A]]#0 : !fir.box<!fir.array<?xi32>>
! CHECK:    } to {
! CHECK:      %[[BOX_B2:.*]] = fir.load %[[B]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:      hlfir.yield %[[BOX_B2]] : !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:    }
          ! ELSEWHERE: b(:) = 0
! CHECK:    hlfir.elsewhere do {
! CHECK:      hlfir.region_assign {
! CHECK:        %[[C0:.*]] = arith.constant 0 : i32
! CHECK:        hlfir.yield %[[C0]] : i32
! CHECK:      } to {
! CHECK:        hlfir.designate %{{.*}} (%{{.*}}:%{{.*}}:%{{.*}})  shape %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:      }
! CHECK:    }
! CHECK:  }
! CHECK:  return
  where (b > 0)
    b = a
  elsewhere
    b(:) = 0
  end where
end
end module

  use mtest
  integer, allocatable :: a(:), b(:)
  allocate(a(10),b(10))
  a = 5
  b = 1
  call foo(a, b)
  print*, b
  deallocate(a,b)
end
