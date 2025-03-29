! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine sub1(x, a)
  real :: x(200)
  integer :: a

  !$acc loop
  do i = 100, 200
    x(i) = 1.0
    if (i == a) return
  end do

  i = 2
end 

! CHECK-LABEL: func.func @_QPsub1
! CHECK: %[[A:.*]]:2 = hlfir.declare %arg1 dummy_scope %{{[0-9]+}} {uniq_name = "_QFsub1Ea"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[I:.*]]:2 = hlfir.declare %{{[0-9]+}} {uniq_name = "_QFsub1Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[I:.*]]:2 = hlfir.declare %{{[0-9]+}} {uniq_name = "_QFsub1Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[EXIT_COND:.*]] = acc.loop
! CHECK: ^bb{{.*}}:
! CHECK: ^bb{{.*}}:
! CHECK:   %[[LOAD_I:.*]] = fir.load %[[I]]#0 : !fir.ref<i32> 
! CHECK:   %[[LOAD_I:.*]] = fir.load %[[I]]#0 : !fir.ref<i32> 
! CHECK:   %[[LOAD_A:.*]] = fir.load %[[A]]#0 : !fir.ref<i32> 
! CHECK:   %[[CMP:.*]] = arith.cmpi eq, %[[LOAD_I]], %[[LOAD_A]] : i32
! CHECK:   cf.cond_br %[[CMP]], ^[[EARLY_RET:.*]], ^[[NO_RET:.*]]
! CHECK: ^[[EARLY_RET]]:
! CHECK:   acc.yield %true : i1
! CHECK: ^[[NO_RET]]:
! CHECK:   cf.br ^bb{{.*}}
! CHECK: ^bb{{.*}}:
! CHECK:   acc.yield %false : i1
! CHECK: }(i1)
! CHECK: cf.cond_br %[[EXIT_COND]], ^[[EXIT_BLOCK:.*]], ^[[CONTINUE_BLOCK:.*]]
! CHECK: ^[[CONTINUE_BLOCK]]:
! CHECK:   hlfir.assign
! CHECK:   cf.br ^[[EXIT_BLOCK]]
! CHECK: ^[[EXIT_BLOCK]]:
! CHECK:   return
! CHECK: }
