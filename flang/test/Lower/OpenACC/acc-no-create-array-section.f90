! This test checks lowering of OpenACC no_create with an array section.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine foo(n)
  real :: a(n)
  !$acc parallel no_create(a(11:20))
    call bar(a)
  !$acc end parallel
end subroutine
! CHECK-LABEL:   func.func @_QPfoo(
! CHECK:           %[[SHAPE_0:.*]] = fir.shape
! CHECK:           %[[BOUNDS_0:.*]] = acc.bounds
! CHECK:           %[[NOCREATE_0:.*]] = acc.nocreate var(%{{.*}} : !fir.box<!fir.array<?xf32>>) bounds(%[[BOUNDS_0]]) -> !fir.box<!fir.array<?xf32>> {name = "a(11:20)"}
! CHECK:           acc.parallel dataOperands(%[[NOCREATE_0]] : !fir.box<!fir.array<?xf32>>) {
! CHECK:             %[[BOX_ADDR_0:.*]] = fir.box_addr %[[NOCREATE_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:             %[[DECLARE_2:.*]]:2 = hlfir.declare %[[BOX_ADDR_0]](%[[SHAPE_0]]) {uniq_name = "_QFfooEa"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:             fir.call @_QPbar(%[[DECLARE_2]]#1) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>) -> ()
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.delete accVar(%[[NOCREATE_0]] : !fir.box<!fir.array<?xf32>>) bounds(%[[BOUNDS_0]]) {dataClause = #acc<data_clause acc_no_create>, name = "a(11:20)"}
