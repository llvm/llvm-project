! An `acc loop seq` with an early EXIT inside an `acc kernels` region. The EXIT
! makes the seq loop and its `acc loop` construct unstructured (!). The seq
! acc.loop privatizes the induction variables of the DOs it owns.
!
! In the first cases, the seq loop is nested in a loop that keeps the branching
! inside its body: that loop and the kernels construct are structured with
! unstructured internals (~), and the loop stays a structured acc.loop with
! bounds. In the next cases, the seq loop is directly in the kernels region, or
! inside an IF, SELECT CASE, BLOCK or DO WHILE there: the constructs around it
! stay unstructured, and their control flow surrounds the seq acc.loop. The last
! case is a combined `kernels loop seq` construct.

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

! A seq loop with an early EXIT directly in the kernels region. Nothing
! encloses it but the kernels construct, which stays unstructured; the seq
! loop's control flow stays in the acc.kernels region.
subroutine seq_exit_in_kernels(m, b, kfirst)
  integer :: m
  real :: b(m)
  integer :: kfirst
  integer :: k
  !$acc kernels
  kfirst = 0
  !$acc loop seq
  do k = 1, m
    if (b(k) > 0.0) then
      kfirst = k
      exit
    end if
  end do
  !$acc end kernels
end subroutine

! PFT-LABEL: Subroutine seq_exit_in_kernels
! PFT:         <<OpenACCConstruct!>>
! PFT:           <<OpenACCConstruct!>>
! PFT-NEXT:        <<DoConstruct!>>
! PFT-NEXT:          NonLabelDoStmt {{.*}}: do k = 1, m
! PFT:               ExitStmt! {{.*}}: exit
! PFT:             <<End DoConstruct!>>
! PFT:           <<End OpenACCConstruct!>>
! PFT:         <<End OpenACCConstruct!>>

! The PFT nodes above lower to:
!   <<OpenACCConstruct!>> kernels  -> acc.kernels
!   <<OpenACCConstruct!>> loop seq -> acc.loop private(k) without bounds
!                                     (seq unstructured)
!   <<DoConstruct!>> do k          -> CFG blocks inside that acc.loop
! CHECK-LABEL: func.func @_QPseq_exit_in_kernels
! CHECK:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFseq_exit_in_kernelsEk")
! CHECK:         acc.kernels {
! CHECK:           %[[K_PRIV:.*]] = acc.private varPtr(%[[K_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK:           acc.loop private(%[[K_PRIV]] : !fir.ref<i32>) {
! CHECK:             hlfir.declare %[[K_PRIV]] uniq_name("_QFseq_exit_in_kernelsEk")
! CHECK:             acc.yield
! CHECK:           } seq  unstructured
! CHECK:           acc.terminator
! CHECK:         }

! The same seq loop followed by a parallelizable loop in the kernels region.
! The following loop stays a structured acc.loop with bounds.
subroutine seq_exit_then_loop(n, m, a, b, kfirst)
  integer :: n, m
  real :: a(n), b(m)
  integer :: kfirst
  integer :: i, k
  !$acc kernels
  kfirst = 0
  !$acc loop seq
  do k = 1, m
    if (b(k) > 0.0) then
      kfirst = k
      exit
    end if
  end do
  do i = 1, n
    a(i) = a(i) + kfirst
  end do
  !$acc end kernels
end subroutine

! PFT-LABEL: Subroutine seq_exit_then_loop
! PFT:         <<OpenACCConstruct!>>
! PFT:           <<OpenACCConstruct!>>
! PFT-NEXT:        <<DoConstruct!>>
! PFT-NEXT:          NonLabelDoStmt {{.*}}: do k = 1, m
! PFT:               ExitStmt! {{.*}}: exit
! PFT:             <<End DoConstruct!>>
! PFT:           <<End OpenACCConstruct!>>
! PFT-NEXT:      <<DoConstruct>>
! PFT-NEXT:        NonLabelDoStmt {{.*}}: do i = 1, n
! PFT:           <<End DoConstruct>>
! PFT:         <<End OpenACCConstruct!>>

! The PFT nodes above lower to:
!   <<OpenACCConstruct!>> kernels  -> acc.kernels
!   <<OpenACCConstruct!>> loop seq -> acc.loop private(k) without bounds
!                                     (seq unstructured)
!   <<DoConstruct!>> do k          -> CFG blocks inside that acc.loop
!   <<DoConstruct>> do i           -> acc.loop with bounds (auto_)
! CHECK-LABEL: func.func @_QPseq_exit_then_loop
! CHECK:         %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFseq_exit_then_loopEi")
! CHECK:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFseq_exit_then_loopEk")
! CHECK:         acc.kernels {
! CHECK:           %[[K_PRIV:.*]] = acc.private varPtr(%[[K_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK:           acc.loop private(%[[K_PRIV]] : !fir.ref<i32>) {
! CHECK:             hlfir.declare %[[K_PRIV]] uniq_name("_QFseq_exit_then_loopEk")
! CHECK:             acc.yield
! CHECK:           } seq  unstructured
! CHECK:           %[[I_PRIV:.*]] = acc.private varPtr(%[[I_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("i")
! CHECK:           acc.loop private(%[[I_PRIV]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32)  step (%{{.*}} : i32) {
! CHECK:             hlfir.declare %[[I_PRIV]] uniq_name("_QFseq_exit_then_loopEi")
! CHECK:             acc.yield
! CHECK:           } inclusiveUpperbound(array<i1: true>) auto_
! CHECK:           acc.terminator
! CHECK:         }

! The same seq loop inside an IF in the kernels region. The IF stays
! unstructured, and its branch surrounds the seq acc.loop.
subroutine seq_exit_in_if(m, b, kfirst, c)
  integer :: m, kfirst
  real :: b(m)
  logical :: c
  integer :: k
  !$acc kernels
  kfirst = 0
  if (c) then
    !$acc loop seq
    do k = 1, m
      if (b(k) > 0.0) then
        kfirst = k
        exit
      end if
    end do
  end if
  !$acc end kernels
end subroutine

! PFT-LABEL: Subroutine seq_exit_in_if
! PFT:         <<OpenACCConstruct>>
! PFT:           <<IfConstruct!>>
! PFT:             <<^OpenACCConstruct!>>
! PFT:               <<DoConstruct!>>
! PFT:                 NonLabelDoStmt {{.*}}: do k = 1, m
! PFT:                   ExitStmt! {{.*}}: exit
! PFT:               <<End DoConstruct!>>
! PFT:             <<End OpenACCConstruct!>>
! PFT:           <<End IfConstruct!>>
! PFT:         <<End OpenACCConstruct>>

! The PFT nodes above lower to:
!   <<OpenACCConstruct>> kernels   -> acc.kernels
!   <<IfConstruct!>>               -> cf.cond_br
!   <<OpenACCConstruct!>> loop seq -> acc.loop private(k) without bounds
!                                     (seq unstructured)
!   <<DoConstruct!>> do k          -> CFG blocks inside that acc.loop
! CHECK-LABEL: func.func @_QPseq_exit_in_if
! CHECK:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFseq_exit_in_ifEk")
! CHECK:         acc.kernels {
! CHECK:           cf.cond_br %{{.*}}, ^[[THEN:.*]], ^{{.*}}
! CHECK:         ^[[THEN]]:
! CHECK:           %[[K_PRIV:.*]] = acc.private varPtr(%[[K_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK:           acc.loop private(%[[K_PRIV]] : !fir.ref<i32>) {
! CHECK:             hlfir.declare %[[K_PRIV]] uniq_name("_QFseq_exit_in_ifEk")
! CHECK:             acc.yield
! CHECK:           } seq  unstructured
! CHECK:           acc.terminator
! CHECK:         }

! The same seq loop inside a SELECT CASE in the kernels region.
subroutine seq_exit_in_select_case(m, b, kfirst, s)
  integer :: m, kfirst
  real :: b(m)
  integer :: s
  integer :: k
  !$acc kernels
  kfirst = 0
  select case (s)
  case (1)
    !$acc loop seq
    do k = 1, m
      if (b(k) > 0.0) then
        kfirst = k
        exit
      end if
    end do
  end select
  !$acc end kernels
end subroutine

! PFT-LABEL: Subroutine seq_exit_in_select_case
! PFT:         <<OpenACCConstruct!>>
! PFT:           <<CaseConstruct!>>
! PFT:             <<OpenACCConstruct!>>
! PFT:               <<DoConstruct!>>
! PFT:                 NonLabelDoStmt {{.*}}: do k = 1, m
! PFT:                   ExitStmt! {{.*}}: exit
! PFT:               <<End DoConstruct!>>
! PFT:             <<End OpenACCConstruct!>>
! PFT:           <<End CaseConstruct!>>
! PFT:         <<End OpenACCConstruct!>>

! The PFT nodes above lower to:
!   <<OpenACCConstruct!>> kernels  -> acc.kernels
!   <<CaseConstruct!>>             -> fir.select_case
!   <<OpenACCConstruct!>> loop seq -> acc.loop private(k) without bounds
!                                     (seq unstructured)
!   <<DoConstruct!>> do k          -> CFG blocks inside that acc.loop
! CHECK-LABEL: func.func @_QPseq_exit_in_select_case
! CHECK:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFseq_exit_in_select_caseEk")
! CHECK:         acc.kernels {
! CHECK:           fir.select_case %{{.*}} : i32 [#fir.point, %{{.*}}, ^[[CASE1:.*]], unit, ^{{.*}}]
! CHECK:         ^[[CASE1]]:
! CHECK:           %[[K_PRIV:.*]] = acc.private varPtr(%[[K_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK:           acc.loop private(%[[K_PRIV]] : !fir.ref<i32>) {
! CHECK:             hlfir.declare %[[K_PRIV]] uniq_name("_QFseq_exit_in_select_caseEk")
! CHECK:             acc.yield
! CHECK:           } seq  unstructured
! CHECK:           acc.terminator
! CHECK:         }

! The same seq loop inside a BLOCK construct in the kernels region.
subroutine seq_exit_in_block(m, b, kfirst)
  integer :: m, kfirst
  real :: b(m)
  integer :: k
  !$acc kernels
  kfirst = 0
  block
    !$acc loop seq
    do k = 1, m
      if (b(k) > 0.0) then
        kfirst = k
        exit
      end if
    end do
  end block
  !$acc end kernels
end subroutine

! PFT-LABEL: Subroutine seq_exit_in_block
! PFT:         <<OpenACCConstruct!>>
! PFT:           <<BlockConstruct!>>
! PFT:             <<OpenACCConstruct!>>
! PFT:               <<DoConstruct!>>
! PFT:                 NonLabelDoStmt {{.*}}: do k = 1, m
! PFT:                   ExitStmt! {{.*}}: exit
! PFT:               <<End DoConstruct!>>
! PFT:             <<End OpenACCConstruct!>>
! PFT:           <<End BlockConstruct!>>
! PFT:         <<End OpenACCConstruct!>>

! The PFT nodes above lower to:
!   <<OpenACCConstruct!>> kernels  -> acc.kernels
!   <<BlockConstruct!>>            -> llvm.intr.stacksave/stackrestore
!   <<OpenACCConstruct!>> loop seq -> acc.loop private(k) without bounds
!                                     (seq unstructured)
!   <<DoConstruct!>> do k          -> CFG blocks inside that acc.loop
! CHECK-LABEL: func.func @_QPseq_exit_in_block
! CHECK:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFseq_exit_in_blockEk")
! CHECK:         acc.kernels {
! CHECK:           llvm.intr.stacksave
! CHECK:           %[[K_PRIV:.*]] = acc.private varPtr(%[[K_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK:           acc.loop private(%[[K_PRIV]] : !fir.ref<i32>) {
! CHECK:             hlfir.declare %[[K_PRIV]] uniq_name("_QFseq_exit_in_blockEk")
! CHECK:             acc.yield
! CHECK:           } seq  unstructured
! CHECK:           acc.terminator
! CHECK:         }

! The same seq loop inside a DO WHILE in the kernels region.
subroutine seq_exit_in_do_while(n, m, b, kfirst)
  integer :: m, kfirst
  real :: b(m)
  integer :: n, i
  integer :: k
  !$acc kernels
  kfirst = 0
  i = 0
  do while (i < n)
    i = i + 1
    !$acc loop seq
    do k = 1, m
      if (b(k) > 0.0) then
        kfirst = k
        exit
      end if
    end do
  end do
  !$acc end kernels
end subroutine

! PFT-LABEL: Subroutine seq_exit_in_do_while
! PFT:         <<OpenACCConstruct!>>
! PFT:           <<DoConstruct!>>
! PFT:             NonLabelDoStmt! {{.*}}: do while(i < n)
! PFT:             <<OpenACCConstruct!>>
! PFT:               <<DoConstruct!>>
! PFT:                 NonLabelDoStmt {{.*}}: do k = 1, m
! PFT:                   ExitStmt! {{.*}}: exit
! PFT:               <<End DoConstruct!>>
! PFT:             <<End OpenACCConstruct!>>
! PFT:           <<End DoConstruct!>>
! PFT:         <<End OpenACCConstruct!>>

! The PFT nodes above lower to:
!   <<OpenACCConstruct!>> kernels  -> acc.kernels
!   <<DoConstruct!>> do while      -> CFG blocks inside acc.kernels
!   <<OpenACCConstruct!>> loop seq -> acc.loop private(k) without bounds
!                                     (seq unstructured)
!   <<DoConstruct!>> do k          -> CFG blocks inside that acc.loop
! CHECK-LABEL: func.func @_QPseq_exit_in_do_while
! CHECK:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFseq_exit_in_do_whileEk")
! CHECK:         acc.kernels {
! CHECK:           cf.cond_br %{{.*}}, ^[[BODY:.*]], ^{{.*}}
! CHECK:         ^[[BODY]]:
! CHECK:           %[[K_PRIV:.*]] = acc.private varPtr(%[[K_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK:           acc.loop private(%[[K_PRIV]] : !fir.ref<i32>) {
! CHECK:             hlfir.declare %[[K_PRIV]] uniq_name("_QFseq_exit_in_do_whileEk")
! CHECK:             acc.yield
! CHECK:           } seq  unstructured
! CHECK:           acc.terminator
! CHECK:         }

! A combined `kernels loop seq` construct with collapse(2), leaving both DOs
! with a named EXIT. The seq acc.loop owns and privatizes both DOs.
subroutine kernels_loop_seq_collapse_exit(m, p, b, kref)
  integer :: m, p
  real :: b(p, m)
  integer :: kref
  integer :: k, l
  kref = 0
  !$acc kernels loop seq collapse(2)
  kloop: do k = 1, m
    do l = 1, p
      if (b(l, k) > 0.0) then
        kref = k
        exit kloop
      end if
    end do
  end do kloop
end subroutine

! PFT-LABEL: Subroutine kernels_loop_seq_collapse_exit
! PFT:         <<OpenACCConstruct!>>
! PFT-NEXT:      <<DoConstruct!>>
! PFT-NEXT:        NonLabelDoStmt {{.*}}: kloop: do k = 1, m
! PFT-NEXT:        <<DoConstruct!>>
! PFT-NEXT:          ^NonLabelDoStmt {{.*}}: do l = 1, p
! PFT:                 ExitStmt! {{.*}}: exit kloop
! PFT:             <<End DoConstruct!>>
! PFT:           <<End DoConstruct!>>
! PFT:         <<End OpenACCConstruct!>>

! The PFT nodes above lower to:
!   <<OpenACCConstruct!>> kernels loop seq -> acc.kernels and an acc.loop
!                                             private(k, l) without bounds
!                                             (seq unstructured)
!   <<DoConstruct!>> do k, do l            -> CFG blocks inside that acc.loop
! CHECK-LABEL: func.func @_QPkernels_loop_seq_collapse_exit
! CHECK:         %[[K_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFkernels_loop_seq_collapse_exitEk")
! CHECK:         %[[L_DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFkernels_loop_seq_collapse_exitEl")
! CHECK:         acc.kernels combined(loop) {
! CHECK:           %[[K_PRIV:.*]] = acc.private varPtr(%[[K_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK:           %[[L_PRIV:.*]] = acc.private varPtr(%[[L_DECL]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("l")
! CHECK:           acc.loop combined(kernels) private(%[[K_PRIV]], %[[L_PRIV]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:             hlfir.declare %[[K_PRIV]] uniq_name("_QFkernels_loop_seq_collapse_exitEk")
! CHECK:             hlfir.declare %[[L_PRIV]] uniq_name("_QFkernels_loop_seq_collapse_exitEl")
! CHECK:             acc.yield
! CHECK:           } collapse([2]) collapseDeviceType([#acc.device_type<none>]) seq  unstructured
! CHECK:           acc.terminator
! CHECK:         }
