! A do concurrent whose branching is confined to its body keeps its structured
! form: its header puts no control flow in the body, so only the body is folded
! into an scf.execute_region. The body checks that apply to an increment loop
! still apply.

! RUN: %flang_fc1 -fdebug-dump-pft -o /dev/null %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s --check-prefix=FIR

! A SELECT CASE is lowered as raw branching, but its branches stay in the body.
subroutine select_case(n, a, tag)
  integer :: n, tag, i
  real :: a(n)
  do concurrent (i = 1:n)
    select case (tag)
    case (1)
      a(i) = 1
    case default
      a(i) = 0
    end select
  end do
end subroutine

! CHECK-LABEL: Subroutine select_case
! CHECK:   <<DoConstruct~>>
! CHECK:     <<CaseConstruct!>>
! CHECK:   <<End DoConstruct~>>

! FIR-LABEL: func.func @_QPselect_case
! FIR:         fir.do_concurrent {
! FIR:           fir.do_concurrent.loop
! FIR:             scf.execute_region no_inline {
! FIR:               fir.select_case
! FIR:               scf.yield
! FIR:             }

! Two indices and a mask: the mask stays a test around the wrapped body.
subroutine mask_two_indices(n, a, tag)
  integer :: n, tag, i, j
  real :: a(n, n)
  do concurrent (i = 1:n, j = 1:n, a(i, j) > 0.0)
    select case (tag)
    case (1)
      a(i, j) = 1
    case default
      a(i, j) = 0
    end select
  end do
end subroutine

! CHECK-LABEL: Subroutine mask_two_indices
! CHECK:   <<DoConstruct~>>
! CHECK:   <<End DoConstruct~>>

! FIR-LABEL: func.func @_QPmask_two_indices
! FIR:         fir.do_concurrent {
! FIR:           fir.do_concurrent.loop (%{{.*}}, %{{.*}}) =
! FIR:             %[[MASK:.*]] = arith.cmpf ogt
! FIR:             fir.if %[[MASK]] {
! FIR:               scf.execute_region no_inline {
! FIR:                 fir.select_case
! FIR:                 scf.yield
! FIR:               }

! A CYCLE targets the EndDoStmt, so it does not escape the body.
subroutine cycle_in_body(n, a, tag)
  integer :: n, tag, i
  real :: a(n)
  do concurrent (i = 1:n)
    select case (tag)
    case (1)
      cycle
    case default
      a(i) = 0
    end select
  end do
end subroutine

! CHECK-LABEL: Subroutine cycle_in_body
! CHECK:   <<DoConstruct~>>
! CHECK:   <<End DoConstruct~>>

! FIR-LABEL: func.func @_QPcycle_in_body
! FIR:         fir.do_concurrent.loop
! FIR:           scf.execute_region no_inline {
! FIR:             fir.select_case
! FIR:             scf.yield
! FIR:           }

! A locality specifier does not change the classification.
subroutine local_spec(n, a, tag)
  integer :: n, tag, i
  real :: a(n), t
  do concurrent (i = 1:n) local(t)
    select case (tag)
    case (1)
      t = 1
    case default
      t = 2
    end select
    a(i) = t
  end do
end subroutine

! CHECK-LABEL: Subroutine local_spec
! CHECK:   <<DoConstruct~>>
! CHECK:   <<End DoConstruct~>>

! FIR-LABEL: func.func @_QPlocal_spec
! FIR:         fir.do_concurrent.loop {{.*}} local(
! FIR:           scf.execute_region no_inline {
! FIR:             fir.select_case
! FIR:             scf.yield
! FIR:           }

! An EXIT of a BLOCK in the body stays in the body.
subroutine exit_block(n, a)
  integer :: n, i
  real :: a(n)
  do concurrent (i = 1:n)
    b: block
      if (a(i) < 0.0) exit b
      a(i) = 1
    end block b
  end do
end subroutine

! CHECK-LABEL: Subroutine exit_block
! CHECK:   <<DoConstruct~>>
! CHECK:     <<BlockConstruct!>>
! CHECK:   <<End DoConstruct~>>

! FIR-LABEL: func.func @_QPexit_block
! FIR:         fir.do_concurrent.loop
! FIR:           scf.execute_region no_inline {
! FIR:             cf.cond_br
! FIR:             scf.yield
! FIR:           }

! An EXIT of an inner increment loop belongs to that loop. The inner loop
! leaves the do concurrent structured, so the do concurrent needs no wrap.
subroutine exit_inner_loop(n, a)
  integer :: n, i, j
  real :: a(n)
  do concurrent (i = 1:n)
    do j = 1, n
      if (a(j) < 0.0) exit
      a(i) = a(i) + 1
    end do
  end do
end subroutine

! CHECK-LABEL: Subroutine exit_inner_loop
! CHECK:   <<DoConstruct>>
! CHECK:     <<DoConstruct!>>
! CHECK:     <<End DoConstruct!>>
! CHECK:   <<End DoConstruct>>

! FIR-LABEL: func.func @_QPexit_inner_loop
! FIR:         fir.do_concurrent.loop
! FIR:           scf.execute_region no_inline {
! FIR:             cf.cond_br
! FIR:             scf.yield
! FIR:           }

! A GOTO back to an earlier statement of the body may never let the body
! complete, so the loop stays unstructured.
subroutine goto_cycle_in_body(n, a)
  integer :: n, i
  real :: a(n)
  do concurrent (i = 1:n)
10  a(i) = a(i) + 1.0
    if (a(i) < 5.0) goto 10
  end do
end subroutine

! CHECK-LABEL: Subroutine goto_cycle_in_body
! CHECK:   <<DoConstruct!>>
! CHECK:   <<End DoConstruct!>>

! The whole construct is folded into one region, with the loop control as raw
! branches inside it.
! FIR-LABEL: func.func @_QPgoto_cycle_in_body
! FIR-NOT:     fir.do_concurrent
! FIR:         scf.execute_region no_inline {
! FIR-NOT:       fir.do_concurrent
! FIR:           cf.cond_br
! FIR:           scf.yield
! FIR:         }

! An infinite DO in the body cannot run to completion either.
subroutine infinite_inner(n, a)
  integer :: n, i
  real :: a(n)
  do concurrent (i = 1:n)
    do
      a(i) = a(i) + 1.0
      if (a(i) > 5.0) exit
    end do
  end do
end subroutine

! CHECK-LABEL: Subroutine infinite_inner
! CHECK:   <<DoConstruct!>>
! CHECK:     <<DoConstruct!>>
! CHECK:     <<End DoConstruct!>>
! CHECK:   <<End DoConstruct!>>

! The infinite DO is not wrappable, so the loop control stays raw branches in
! the function body.
! FIR-LABEL: func.func @_QPinfinite_inner
! FIR-NOT:     fir.do_concurrent
! FIR-NOT:     scf.execute_region
! FIR:         cf.cond_br
! FIR:         return
