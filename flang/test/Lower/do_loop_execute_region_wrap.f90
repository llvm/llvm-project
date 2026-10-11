! RUN: bbc -emit-hlfir --wrap-unstructured-constructs-in-execute-region %s -o - | FileCheck %s

! An unstructured IF inside a DO is self-contained, so isUnstructured does
! not propagate to the DO. The DO lowers as fir.do_loop and the IF's blocks
! are wrapped in an scf.execute_region inside its body.
subroutine wrapped_unstructured(n, a)
  integer :: n, i
  real :: a(n)
  do i = 1, n
    if (a(i) > 0.0) stop
    a(i) = real(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPwrapped_unstructured
! CHECK:         fir.do_loop
! CHECK:           scf.execute_region no_inline {
! CHECK:             cf.cond_br
! CHECK:             fir.call @_FortranAStopStatement
! CHECK:             fir.unreachable
! CHECK:             scf.yield
! CHECK:           }

! Unstructured DO with a GOTO targeting a label outside the construct that
! is reachable on a path distinct from the loop's natural exit: not wrapped.
! The GOTO jumps past the post-loop store directly to label 99, so the
! loop body has an outgoing edge that is neither another loop block nor the
! construct exit. The wrap pass bails out and the CFG stays flat.
subroutine not_wrapped_outer_label(n, a)
  integer :: n, i
  real :: a(n)
  do i = 1, n
    if (a(i) > 0.0) goto 99
    a(i) = real(i)
  end do
  a(1) = -1.0
99 continue
end subroutine

! CHECK-LABEL: func.func @_QPnot_wrapped_outer_label
! CHECK-NOT:     scf.execute_region
! CHECK:         cf.cond_br
! CHECK:         return

! Multiway branch (computed GO TO) inside a DO with a target that escapes
! the loop: not wrapped. Only the first target of a computed GO TO used to
! be considered by the wrappability check, so an escaping non-first label
! (99 below) was invisible and the DO was wrapped; the resulting
! scf.execute_region contained a fir.select whose successor lived outside
! the region, tripping MLIR's op verifier.
subroutine not_wrapped_computed_goto_exit(sel, a)
  integer :: sel, a, i
  do i = 1, 10
    go to (10, 99), sel
10 end do
  a = -1
99 continue
end subroutine

! CHECK-LABEL: func.func @_QPnot_wrapped_computed_goto_exit
! CHECK-NOT:     scf.execute_region
! CHECK:         fir.select
! CHECK:         return

! Same shape via an arithmetic IF. Arithmetic IF has three label targets;
! only the first was recorded on the source Evaluation, so a non-first
! escape target (99) was again invisible to the wrappability check.
subroutine not_wrapped_arithmetic_if_exit(x, a)
  integer :: a, i
  real :: x
  do i = 1, 10
    if (x) 10, 10, 99
10 end do
  a = -1
99 continue
end subroutine

! CHECK-LABEL: func.func @_QPnot_wrapped_arithmetic_if_exit
! CHECK-NOT:     scf.execute_region
! CHECK:         cf.cond_br
! CHECK:         return

! Assigned GO TO with no explicit label list inside a DO: not wrapped.
! The set of runtime targets for `go to k` is the union of every label
! ASSIGN'd to k, but analyzeBranches only sees ASSIGNs earlier than the
! goto in program order.  Here the escaping label 99 is added by a later
! ASSIGN and would be invisible to the escape check, so wrappability bails
! out for any listless assigned GO TO regardless of the currently recorded
! targets.
subroutine not_wrapped_assigned_goto_no_list(a)
  integer :: a, i, k
  assign 10 to k
  do i = 1, 10
    go to k
    assign 99 to k
10 end do
  a = -1
99 continue
end subroutine

! CHECK-LABEL: func.func @_QPnot_wrapped_assigned_goto_no_list
! CHECK-NOT:     scf.execute_region
! CHECK:         return

! A plain, structured DO with no early exits: lowered as fir.do_loop,
! never reaches the wrap path (the loop is not unstructured at all).
subroutine structured(n, a)
  integer :: n, i
  real :: a(n)
  do i = 1, n
    a(i) = real(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPstructured
! CHECK-NOT:     scf.execute_region
! CHECK:         fir.do_loop
! CHECK:         return

! Nested DOs whose only unstructuredness comes from a self-contained IF.
! Both DOs lower as fir.do_loop; only the IF is wrapped in scf.execute_region
! at the innermost level.
subroutine outer_structured_inner_wrapped(n, a)
  integer :: n, i, j
  real :: a(n)
  do i = 1, n
    do j = 1, n
      if (a(j) > 0.0) stop
      a(j) = real(i + j)
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPouter_structured_inner_wrapped
! CHECK:         fir.do_loop
! CHECK:           fir.do_loop
! CHECK:             scf.execute_region no_inline {
! CHECK:               cf.cond_br
! CHECK:               fir.call @_FortranAStopStatement
! CHECK:               fir.unreachable
! CHECK:               scf.yield
! CHECK:             }

! Structured outer DO containing an unstructured inner DO (the inner DO has
! an EXIT that targets its own construct exit). isUnstructured does not
! propagate from the inner DO to the outer DO, so the outer lowers as
! fir.do_loop. The inner DO's blocks would otherwise have nowhere to live
! inside the outer's single-block region, so the pre-wrap path creates an
! scf.execute_region around the inner DO's CFG.
subroutine outer_fir_do_loop_inner_unstructured_do(n, a)
  integer :: n, i, j
  real :: a(n)
  do i = 1, n
    do j = 1, n
      if (a(j) > 0.0) exit
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPouter_fir_do_loop_inner_unstructured_do
! CHECK:         fir.do_loop
! CHECK:           scf.execute_region no_inline {
! CHECK:             cf.br
! CHECK:             cf.cond_br
! CHECK:             cf.cond_br
! CHECK:             scf.yield
! CHECK:           }
