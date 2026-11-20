! A DO WHILE after an `acc loop seq` directive. A DO WHILE is always lowered
! with explicit control flow, so the seq loop and its `acc loop` construct are
! unstructured (!): the acc.loop carries no bounds and holds the loop's blocks.
! A DO WHILE has no induction variable, so nothing is privatized for it.
!
! With -lower-do-while-to-scf-while, a DO WHILE without branches out of it stays
! structured: the acc.loop, still without bounds, holds an scf.while.

! RUN: bbc -fopenacc -pft-test -o /dev/null %s | FileCheck %s --check-prefix=PFT
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenacc -lower-do-while-to-scf-while -pft-test -o /dev/null %s | FileCheck %s --check-prefix=PFT-SCF
! RUN: bbc -fopenacc -lower-do-while-to-scf-while -emit-hlfir %s -o - | FileCheck %s --check-prefix=SCF

subroutine do_while_in_kernels(n, a, k)
  integer :: n, k
  real :: a(n)
  !$acc kernels
  k = 0
  !$acc loop seq
  do while (k < n)
    k = k + 1
    a(k) = 1.0
  end do
  !$acc end kernels
end subroutine

! PFT-LABEL: Subroutine do_while_in_kernels
! PFT:         <<OpenACCConstruct!>>
! PFT:           <<OpenACCConstruct!>>
! PFT-NEXT:        <<DoConstruct!>>
! PFT-NEXT:          NonLabelDoStmt! {{.*}}: do while(k < n)
! PFT:             <<End DoConstruct!>>
! PFT:           <<End OpenACCConstruct!>>
! PFT:         <<End OpenACCConstruct!>>

! The PFT nodes above lower to:
!   <<OpenACCConstruct!>> kernels  -> acc.kernels
!   <<OpenACCConstruct!>> loop seq -> acc.loop without bounds or private
!                                     operands (seq unstructured)
!   <<DoConstruct!>> do while      -> CFG blocks inside that acc.loop
! CHECK-LABEL: func.func @_QPdo_while_in_kernels
! CHECK:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdo_while_in_kernelsEk")
! CHECK:         acc.kernels {
! CHECK:           acc.loop {
! CHECK:           ^bb1:
! CHECK:             fir.load %[[K_DECL]]#0
! CHECK:             cf.cond_br %{{.*}}, ^bb2, ^bb3
! CHECK:           ^bb3:
! CHECK:             acc.yield
! CHECK:           } seq  unstructured
! CHECK:           acc.terminator
! CHECK:         }

! PFT-SCF-LABEL: Subroutine do_while_in_kernels
! PFT-SCF:         <<OpenACCConstruct>>
! PFT-SCF:           <<OpenACCConstruct>>
! PFT-SCF-NEXT:        <<DoConstruct>>
! PFT-SCF-NEXT:          NonLabelDoStmt {{.*}}: do while(k < n)
! PFT-SCF:             <<End DoConstruct>>
! PFT-SCF:           <<End OpenACCConstruct>>
! PFT-SCF:         <<End OpenACCConstruct>>

! SCF-LABEL: func.func @_QPdo_while_in_kernels
! SCF:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdo_while_in_kernelsEk")
! SCF:         acc.kernels {
! SCF:           acc.loop {
! SCF:             scf.while : () -> () {
! SCF:               fir.load %[[K_DECL]]#0
! SCF:               scf.condition(%{{.*}})
! SCF:             } do {
! SCF:               scf.yield
! SCF:             }
! SCF:             acc.yield
! SCF:           } seq  unstructured
! SCF:           acc.terminator
! SCF:         }


! The same DO WHILE after a combined `parallel loop seq` directive.
subroutine do_while_parallel_loop(n, a, k)
  integer :: n, k
  real :: a(n)
  k = 0
  !$acc parallel loop seq
  do while (k < n)
    k = k + 1
    a(k) = 1.0
  end do
end subroutine

! PFT-LABEL: Subroutine do_while_parallel_loop
! PFT:         <<OpenACCConstruct!>>
! PFT-NEXT:      <<DoConstruct!>>
! PFT-NEXT:        NonLabelDoStmt! {{.*}}: do while(k < n)
! PFT:           <<End DoConstruct!>>
! PFT:         <<End OpenACCConstruct!>>

! The PFT nodes above lower to:
!   <<OpenACCConstruct!>> parallel loop seq -> acc.parallel holding an acc.loop
!                                              without bounds (seq unstructured)
!   <<DoConstruct!>> do while               -> CFG blocks inside that acc.loop
! CHECK-LABEL: func.func @_QPdo_while_parallel_loop
! CHECK:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdo_while_parallel_loopEk")
! CHECK:         acc.parallel combined(loop) {
! CHECK:           acc.loop combined(parallel) {
! CHECK:           ^bb1:
! CHECK:             fir.load %[[K_DECL]]#0
! CHECK:             cf.cond_br %{{.*}}, ^bb2, ^bb3
! CHECK:           ^bb3:
! CHECK:             acc.yield
! CHECK:           } seq  unstructured
! CHECK:           acc.yield
! CHECK:         }

! PFT-SCF-LABEL: Subroutine do_while_parallel_loop
! PFT-SCF:         <<OpenACCConstruct>>
! PFT-SCF-NEXT:      <<DoConstruct>>
! PFT-SCF-NEXT:        NonLabelDoStmt {{.*}}: do while(k < n)
! PFT-SCF:           <<End DoConstruct>>
! PFT-SCF:         <<End OpenACCConstruct>>

! SCF-LABEL: func.func @_QPdo_while_parallel_loop
! SCF:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdo_while_parallel_loopEk")
! SCF:         acc.parallel combined(loop) {
! SCF:           acc.loop combined(parallel) {
! SCF:             scf.while : () -> () {
! SCF:               fir.load %[[K_DECL]]#0
! SCF:               scf.condition(%{{.*}})
! SCF:             } do {
! SCF:               scf.yield
! SCF:             }
! SCF:             acc.yield
! SCF:           } seq  unstructured
! SCF:           acc.yield
! SCF:         }


! A DO WHILE seq loop nested in a loop in kernels: the enclosing loop stays a
! structured acc.loop with bounds.
subroutine do_while_in_loops(n, m, b, kref)
  integer :: n, m
  real :: b(m, n)
  integer :: kref(n)
  integer :: i, k
  !$acc kernels
  do i = 1, n
    k = 1
    !$acc loop seq
    do while (k <= m .and. b(min(k, m), i) <= 0.0)
      k = k + 1
    end do
    kref(i) = k
  end do
  !$acc end kernels
end subroutine

! PFT-LABEL: Subroutine do_while_in_loops
! PFT:         <<OpenACCConstruct~>>
! PFT-NEXT:      <<DoConstruct~>>
! PFT-NEXT:        NonLabelDoStmt {{.*}}: do i = 1, n
! PFT:             <<OpenACCConstruct!>>
! PFT-NEXT:          <<DoConstruct!>>
! PFT-NEXT:            NonLabelDoStmt! {{.*}}: do while(k <= m .and. b(min(k, m), i) <= 0.0)
! PFT:               <<End DoConstruct!>>
! PFT:             <<End OpenACCConstruct!>>
! PFT:           <<End DoConstruct~>>
! PFT:         <<End OpenACCConstruct~>>

! The PFT nodes above lower to:
!   <<OpenACCConstruct~>> kernels  -> acc.kernels
!   <<DoConstruct~>> do i          -> acc.loop with bounds (auto_), its body in
!                                     an scf.execute_region
!   <<OpenACCConstruct!>> loop seq -> acc.loop without bounds or private
!                                     operands (seq unstructured)
!   <<DoConstruct!>> do while      -> CFG blocks inside that acc.loop
! CHECK-LABEL: func.func @_QPdo_while_in_loops
! CHECK:         %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdo_while_in_loopsEi")
! CHECK:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdo_while_in_loopsEk")
! CHECK:         acc.kernels {
! CHECK:           %[[I_PRIV:.*]] = acc.private varPtr(%[[I_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("i")
! CHECK:           acc.loop private(%[[I_PRIV]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32)  step (%{{.*}} : i32) {
! CHECK:             scf.execute_region no_inline {
! CHECK:               acc.loop {
! CHECK:               ^bb1:
! CHECK:                 fir.load %[[K_DECL]]#0
! CHECK:                 cf.cond_br %{{.*}}, ^bb2, ^bb3
! CHECK:               ^bb3:
! CHECK:                 acc.yield
! CHECK:               } seq  unstructured
! CHECK:               scf.yield
! CHECK:             }
! CHECK:             acc.yield
! CHECK:           } inclusiveUpperbound(array<i1: true>) auto_
! CHECK:           acc.terminator
! CHECK:         }

! PFT-SCF-LABEL: Subroutine do_while_in_loops
! PFT-SCF:         <<OpenACCConstruct>>
! PFT-SCF-NEXT:      <<DoConstruct>>
! PFT-SCF-NEXT:        NonLabelDoStmt {{.*}}: do i = 1, n
! PFT-SCF:             <<OpenACCConstruct>>
! PFT-SCF-NEXT:          <<DoConstruct>>
! PFT-SCF-NEXT:            NonLabelDoStmt {{.*}}: do while(k <= m .and. b(min(k, m), i) <= 0.0)
! PFT-SCF:               <<End DoConstruct>>
! PFT-SCF:             <<End OpenACCConstruct>>
! PFT-SCF:           <<End DoConstruct>>
! PFT-SCF:         <<End OpenACCConstruct>>

! SCF-LABEL: func.func @_QPdo_while_in_loops
! SCF:         %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdo_while_in_loopsEi")
! SCF:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdo_while_in_loopsEk")
! SCF:         acc.kernels {
! SCF:           %[[I_PRIV:.*]] = acc.private varPtr(%[[I_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("i")
! SCF:           acc.loop private(%[[I_PRIV]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32)  step (%{{.*}} : i32) {
! SCF:             acc.loop {
! SCF:               scf.while : () -> () {
! SCF:                 fir.load %[[K_DECL]]#0
! SCF:                 scf.condition(%{{.*}})
! SCF:               } do {
! SCF:                 scf.yield
! SCF:               }
! SCF:               acc.yield
! SCF:             } seq  unstructured
! SCF:             acc.yield
! SCF:           } inclusiveUpperbound(array<i1: true>) auto_
! SCF:           acc.terminator
! SCF:         }
