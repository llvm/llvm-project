! An `acc loop seq` with an early EXIT inside an `acc kernels` region. The EXIT
! makes the seq loop unstructured, but its enclosing loop keeps its branching
! inside its body, so the seq loop's control flow is contained there: the
! enclosing loop stays a structured acc.loop with bounds, and the seq acc.loop
! privatizes the induction variables of the DOs it owns.
!
! In the pre-FIR tree, the seq loop and its `acc loop` construct are
! unstructured (!), while the enclosing loop and the kernels construct are
! structured with unstructured internals (~).

! RUN: bbc -fopenacc -pft-test -o /dev/null %s | FileCheck %s --check-prefix=PFT
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine seq_exit_in_loop(n, m, b, kref)
  integer :: n, m
  real :: b(m, n)
  integer :: kref(n)
  integer :: i, k
  !$acc kernels
  do i = 1, n
    kref(i) = 0
    !$acc loop seq
    do k = 1, m
      if (b(k, i) > 0.0) then
        kref(i) = k
        exit
      end if
    end do
  end do
  !$acc end kernels
end subroutine

! PFT-LABEL: Subroutine seq_exit_in_loop
! PFT:         <<OpenACCConstruct~>>
! PFT-NEXT:      <<DoConstruct~>>
! PFT-NEXT:        NonLabelDoStmt {{.*}}: do i = 1, n
! PFT:             <<OpenACCConstruct!>>
! PFT-NEXT:          <<DoConstruct!>>
! PFT-NEXT:            NonLabelDoStmt {{.*}}: do k = 1, m
! PFT:                 ExitStmt! {{.*}}: exit
! PFT:               <<End DoConstruct!>>
! PFT:             <<End OpenACCConstruct!>>
! PFT:           <<End DoConstruct~>>
! PFT:         <<End OpenACCConstruct~>>

! The PFT nodes above lower to:
!   <<OpenACCConstruct~>> kernels  -> acc.kernels
!   <<DoConstruct~>> do i          -> acc.loop with bounds (auto_), its body in
!                                     an scf.execute_region
!   <<OpenACCConstruct!>> loop seq -> acc.loop private(k) without bounds
!                                     (seq unstructured)
!   <<DoConstruct!>> do k          -> CFG blocks inside that acc.loop
! CHECK-LABEL: func.func @_QPseq_exit_in_loop
! CHECK:         %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFseq_exit_in_loopEi")
! CHECK:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFseq_exit_in_loopEk")
! CHECK:         acc.kernels {
! CHECK:           %[[I_PRIV:.*]] = acc.private varPtr(%[[I_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("i")
! CHECK:           acc.loop private(%[[I_PRIV]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32)  step (%{{.*}} : i32) {
! CHECK:             hlfir.declare %[[I_PRIV]] uniq_name("_QFseq_exit_in_loopEi")
! CHECK:             scf.execute_region no_inline {
! CHECK:               %[[K_PRIV:.*]] = acc.private varPtr(%[[K_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK:               acc.loop private(%[[K_PRIV]] : !fir.ref<i32>) {
! CHECK:                 hlfir.declare %[[K_PRIV]] uniq_name("_QFseq_exit_in_loopEk")
! CHECK:                 acc.yield
! CHECK:               } seq  unstructured
! CHECK:               scf.yield
! CHECK:             }
! CHECK:             acc.yield
! CHECK:           } inclusiveUpperbound(array<i1: true>) auto_
! CHECK:           acc.terminator
! CHECK:         }

! With collapse(2), the inner DO belongs to the directive too, so the seq
! acc.loop privatizes both induction variables.
subroutine seq_collapse_exit_in_loop(n, m, p, b, kref)
  integer :: n, m, p
  real :: b(p, m, n)
  integer :: kref(n)
  integer :: i, k, l
  !$acc kernels
  do i = 1, n
    kref(i) = 0
    !$acc loop seq collapse(2)
    kloop: do k = 1, m
      do l = 1, p
        if (b(l, k, i) > 0.0) then
          kref(i) = k
          exit kloop
        end if
      end do
    end do kloop
  end do
  !$acc end kernels
end subroutine

! PFT-LABEL: Subroutine seq_collapse_exit_in_loop
! PFT:         <<OpenACCConstruct~>>
! PFT-NEXT:      <<DoConstruct~>>
! PFT-NEXT:        NonLabelDoStmt {{.*}}: do i = 1, n
! PFT:             <<OpenACCConstruct!>>
! PFT-NEXT:          <<DoConstruct!>>
! PFT-NEXT:            NonLabelDoStmt {{.*}}: kloop: do k = 1, m
! PFT-NEXT:            <<DoConstruct!>>
! PFT-NEXT:              NonLabelDoStmt {{.*}}: do l = 1, p
! PFT:                   ExitStmt! {{.*}}: exit kloop
! PFT:                 <<End DoConstruct!>>
! PFT:               <<End DoConstruct!>>
! PFT:             <<End OpenACCConstruct!>>
! PFT:           <<End DoConstruct~>>
! PFT:         <<End OpenACCConstruct~>>

! The PFT nodes above lower to:
!   <<OpenACCConstruct~>> kernels  -> acc.kernels
!   <<DoConstruct~>> do i          -> acc.loop with bounds (auto_), its body in
!                                     an scf.execute_region
!   <<OpenACCConstruct!>> loop seq -> acc.loop private(k, l) without bounds
!                                     (collapse 2, seq unstructured)
!   <<DoConstruct!>> do k, do l    -> CFG blocks inside that acc.loop
! CHECK-LABEL: func.func @_QPseq_collapse_exit_in_loop
! CHECK:         %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFseq_collapse_exit_in_loopEi")
! CHECK:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFseq_collapse_exit_in_loopEk")
! CHECK:         %[[L_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFseq_collapse_exit_in_loopEl")
! CHECK:         acc.kernels {
! CHECK:           %[[I_PRIV:.*]] = acc.private varPtr(%[[I_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("i")
! CHECK:           acc.loop private(%[[I_PRIV]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32)  step (%{{.*}} : i32) {
! CHECK:             hlfir.declare %[[I_PRIV]] uniq_name("_QFseq_collapse_exit_in_loopEi")
! CHECK:             scf.execute_region no_inline {
! CHECK:               %[[K_PRIV:.*]] = acc.private varPtr(%[[K_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK:               %[[L_PRIV:.*]] = acc.private varPtr(%[[L_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("l")
! CHECK:               acc.loop private(%[[K_PRIV]], %[[L_PRIV]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:                 hlfir.declare %[[K_PRIV]] uniq_name("_QFseq_collapse_exit_in_loopEk")
! CHECK:                 hlfir.declare %[[L_PRIV]] uniq_name("_QFseq_collapse_exit_in_loopEl")
! CHECK:                 acc.yield
! CHECK:               } collapse([2]) collapseDeviceType([#acc.device_type<none>]) seq  unstructured
! CHECK:               scf.yield
! CHECK:             }
! CHECK:             acc.yield
! CHECK:           } inclusiveUpperbound(array<i1: true>) auto_
! CHECK:           acc.terminator
! CHECK:         }
