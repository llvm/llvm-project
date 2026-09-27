! RUN: bbc -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -emit-hlfir -fwrapv %s -o - | FileCheck %s --check-prefix=NO-NSW

! NO-NSW-NOT: overflow<nsw>

! CHECK-LABEL: func @_QPtest(
! CHECK-SAME:     %[[arg0:.*]]: !fir.boxchar<1>{{.*}}) -> !fir.array<1x!fir.logical<4>> {
! CHECK:         %[[unbox:.*]]:2 = fir.unboxchar %[[arg0]]
! CHECK:         %[[addr:.*]] = fir.convert %[[unbox]]#0
! CHECK:         %[[c:.*]]:2 = hlfir.declare %[[addr]](%{{.*}}) typeparams %{{.*}}
! CHECK:         %[[test:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtestEtest"}
! CHECK:         %[[slice:.*]] = hlfir.designate %[[c]]#0 (%c1{{.*}}:%c1{{.*}}:%c1{{.*}}) substr %c1{{.*}}, %c8{{.*}}
! CHECK:         %[[const:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.1x8xc1.0"}
! CHECK:         %[[res:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<1x!fir.logical<4>> {
! CHECK:         ^bb0(%[[idx:.*]]: index):
! CHECK:           %[[lhs_addr:.*]] = hlfir.designate %[[slice]] (%[[idx]])
! CHECK:           %[[rhs_addr:.*]] = hlfir.designate %[[const]]#0 (%[[idx]])
! CHECK:           %[[cmp:.*]] = hlfir.cmpchar eq %[[lhs_addr]] %[[rhs_addr]]
! CHECK:           %[[cast:.*]] = fir.convert %[[cmp]]
! CHECK:           hlfir.yield_element %[[cast]]
! CHECK:         }
! CHECK:         hlfir.assign %[[res]] to %[[test]]#0
! CHECK:         %[[ret:.*]] = fir.load %[[test]]#0
! CHECK:         return %[[ret]]

function test(C)
  logical :: test(1)
  character*12  C(1)

  test = C(1:1)(1:8) == (/'ABCDabcd'/)
end function test
