! An `acc loop seq` with an early EXIT inside an `acc kernels` region. The EXIT
! makes the seq loop unstructured, but its enclosing loop keeps its branching
! inside its body, so the seq loop's control flow is contained there: the
! enclosing loop stays a structured acc.loop with bounds, and the seq acc.loop
! privatizes the induction variables of the DOs it owns and of the
! unstructured DOs nested in its body.
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

! A plain DO in the body of the seq loop, leaving the seq loop with a named
! EXIT. The seq loop does not own the plain DO, but the DO runs on the thread
! executing the seq loop, so the seq acc.loop privatizes its induction variable
! too.
subroutine plain_do_in_seq_loop(n, m, p, b, kref)
  integer :: n, m, p
  real :: b(p, m, n)
  integer :: kref(n)
  integer :: i, k, l
  !$acc kernels
  do i = 1, n
    kref(i) = 0
    !$acc loop seq
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

! PFT-LABEL: Subroutine plain_do_in_seq_loop
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
!                                     (seq unstructured)
!   <<DoConstruct!>> do k, do l    -> CFG blocks inside that acc.loop
! CHECK-LABEL: func.func @_QPplain_do_in_seq_loop
! CHECK:         %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFplain_do_in_seq_loopEi")
! CHECK:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFplain_do_in_seq_loopEk")
! CHECK:         %[[L_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFplain_do_in_seq_loopEl")
! CHECK:         acc.kernels {
! CHECK:           %[[I_PRIV:.*]] = acc.private varPtr(%[[I_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("i")
! CHECK:           acc.loop private(%[[I_PRIV]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32)  step (%{{.*}} : i32) {
! CHECK:             hlfir.declare %[[I_PRIV]] uniq_name("_QFplain_do_in_seq_loopEi")
! CHECK:             scf.execute_region no_inline {
! CHECK:               %[[K_PRIV:.*]] = acc.private varPtr(%[[K_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK:               %[[L_PRIV:.*]] = acc.private varPtr(%[[L_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("l")
! CHECK:               acc.loop private(%[[K_PRIV]], %[[L_PRIV]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:                 hlfir.declare %[[K_PRIV]] uniq_name("_QFplain_do_in_seq_loopEk")
! CHECK:                 hlfir.declare %[[L_PRIV]] uniq_name("_QFplain_do_in_seq_loopEl")
! CHECK:                 acc.yield
! CHECK:               } seq  unstructured
! CHECK:               scf.yield
! CHECK:             }
! CHECK:             acc.yield
! CHECK:           } inclusiveUpperbound(array<i1: true>) auto_
! CHECK:           acc.terminator
! CHECK:         }

! collapse(force:2) absorbs only the first DO nested in the DO over k, the one
! over l. The DO over q next to it is a plain DO of the seq loop's body, so the
! seq acc.loop privatizes k, l and q. The DO over l is structured and also gets
! an acc.loop of its own.
subroutine collapse_force_sibling_do(n, m, p, b, kref)
  integer :: n, m, p
  real :: b(p, m, n)
  integer :: kref(n)
  integer :: i, k, l, q
  !$acc kernels
  do i = 1, n
    kref(i) = 0
    !$acc loop seq collapse(force:2)
    kloop: do k = 1, m
      do l = 1, p
        kref(i) = kref(i) + 1
      end do
      do q = 1, p
        if (b(q, k, i) > 0.0) then
          kref(i) = k
          exit kloop
        end if
      end do
    end do kloop
  end do
  !$acc end kernels
end subroutine

! PFT-LABEL: Subroutine collapse_force_sibling_do
! PFT:         <<OpenACCConstruct~>>
! PFT-NEXT:      <<DoConstruct~>>
! PFT-NEXT:        NonLabelDoStmt {{.*}}: do i = 1, n
! PFT:             <<OpenACCConstruct!>>
! PFT-NEXT:          <<DoConstruct!>>
! PFT-NEXT:            NonLabelDoStmt {{.*}}: kloop: do k = 1, m
! PFT-NEXT:            <<DoConstruct>>
! PFT-NEXT:              NonLabelDoStmt {{.*}}: do l = 1, p
! PFT:                 <<End DoConstruct>>
! PFT-NEXT:            <<DoConstruct!>>
! PFT-NEXT:              NonLabelDoStmt {{.*}}: do q = 1, p
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
!   <<OpenACCConstruct!>> loop seq -> acc.loop private(k, l, q) without bounds
!                                     (collapse 2, seq unstructured)
!   <<DoConstruct!>> do k, do q    -> CFG blocks inside that acc.loop
!   <<DoConstruct>> do l           -> acc.loop with bounds (auto_) inside that
!                                     acc.loop
! CHECK-LABEL: func.func @_QPcollapse_force_sibling_do
! CHECK:         %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFcollapse_force_sibling_doEi")
! CHECK:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFcollapse_force_sibling_doEk")
! CHECK:         %[[L_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFcollapse_force_sibling_doEl")
! CHECK:         %[[Q_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFcollapse_force_sibling_doEq")
! CHECK:         acc.kernels {
! CHECK:           %[[I_PRIV:.*]] = acc.private varPtr(%[[I_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("i")
! CHECK:           acc.loop private(%[[I_PRIV]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32)  step (%{{.*}} : i32) {
! CHECK:             hlfir.declare %[[I_PRIV]] uniq_name("_QFcollapse_force_sibling_doEi")
! CHECK:             scf.execute_region no_inline {
! CHECK:               %[[K_PRIV:.*]] = acc.private varPtr(%[[K_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK:               %[[L_PRIV:.*]] = acc.private varPtr(%[[L_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("l")
! CHECK:               %[[Q_PRIV:.*]] = acc.private varPtr(%[[Q_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("q")
! CHECK:               acc.loop private(%[[K_PRIV]], %[[L_PRIV]], %[[Q_PRIV]] : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:                 hlfir.declare %[[K_PRIV]] uniq_name("_QFcollapse_force_sibling_doEk")
! CHECK:                 %[[L_SEQ:.*]]:2 = hlfir.declare %[[L_PRIV]] uniq_name("_QFcollapse_force_sibling_doEl")
! CHECK:                 hlfir.declare %[[Q_PRIV]] uniq_name("_QFcollapse_force_sibling_doEq")
! CHECK:                 %[[L_INNER:.*]] = acc.private varPtr(%[[L_SEQ]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("l")
! CHECK:                 acc.loop private(%[[L_INNER]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32)  step (%{{.*}} : i32) {
! CHECK:                   hlfir.declare %[[L_INNER]] uniq_name("_QFcollapse_force_sibling_doEl")
! CHECK:                   acc.yield
! CHECK:                 } inclusiveUpperbound(array<i1: true>) auto_
! CHECK:                 acc.yield
! CHECK:               } collapse([2]) collapseDeviceType([#acc.device_type<none>]) seq  unstructured
! CHECK:               scf.yield
! CHECK:             }
! CHECK:             acc.yield
! CHECK:           } inclusiveUpperbound(array<i1: true>) auto_
! CHECK:           acc.terminator
! CHECK:         }
