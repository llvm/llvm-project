! Test lowering of deviceptr clause.
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine test (a, b, n)
   real(8) :: a(:), b(:)
   !$acc parallel loop deviceptr(a,b)
   do i = 1,n
      a(i) = b(i)
   enddo
end subroutine
! CHECK-LABEL:   func.func @_QPtest(
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare {{.*}}"_QFtestEa"
! CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare {{.*}}"_QFtestEb"
! CHECK:           %[[DEVICEPTR_0:.*]] = acc.deviceptr var(%[[DECLARE_0]]#0 : !fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>> {name = "a"}
! CHECK:           %[[DEVICEPTR_1:.*]] = acc.deviceptr var(%[[DECLARE_1]]#0 : !fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>> {name = "b"}
! CHECK:           acc.parallel combined(loop) dataOperands(%[[DEVICEPTR_0]], %[[DEVICEPTR_1]] : !fir.box<!fir.array<?xf64>>, !fir.box<!fir.array<?xf64>>) {
! CHECK:             %[[DECLARE_4:.*]]:2 = hlfir.declare %[[DEVICEPTR_0]]
! CHECK:             %[[DECLARE_5:.*]]:2 = hlfir.declare %[[DEVICEPTR_1]]
! CHECK:             acc.loop combined(parallel)
! CHECK:               hlfir.designate %[[DECLARE_5]]#0
! CHECK:               hlfir.designate %[[DECLARE_4]]#0
