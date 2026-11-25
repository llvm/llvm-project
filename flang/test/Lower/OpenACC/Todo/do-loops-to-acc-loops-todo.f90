! RUN: split-file --leading-lines %s %t
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/do_loop_with_cycle_goto.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK2
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/nested_goto_loop.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK3
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/collapse_lt.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK7
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/collapse_gt.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK8
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/collapse_nested.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK6
! RUN: bbc -fopenacc -pft-test -o /dev/null %t/plain_do_in_seq_loop.f90 | FileCheck %s --check-prefix=PFT10
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/plain_do_in_seq_loop.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK10
! RUN: bbc -fopenacc -pft-test -o /dev/null %t/collapse_force_sibling_do.f90 | FileCheck %s --check-prefix=PFT11
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/collapse_force_sibling_do.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK11
! RUN: bbc -fopenacc -pft-test -o /dev/null %t/seq_exit_device_type_seq.f90 | FileCheck %s --check-prefix=PFT16
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/seq_exit_device_type_seq.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK16
! RUN: bbc -fopenacc -pft-test -o /dev/null %t/auto_exit_in_kernels.f90 | FileCheck %s --check-prefix=PFT17
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/auto_exit_in_kernels.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK17
! RUN: bbc -fopenacc -pft-test -o /dev/null %t/loop_exit_in_kernels.f90 | FileCheck %s --check-prefix=PFT18
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/loop_exit_in_kernels.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK18
! RUN: bbc -fopenacc -pft-test -o /dev/null %t/collapse_force_sibling_same_var.f90 | FileCheck %s --check-prefix=PFT19
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/collapse_force_sibling_same_var.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK19

//--- do_loop_with_cycle_goto.f90

subroutine do_loop_with_cycle_goto()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  ! Do loop with cycle and goto - unstructured control flow is not converted.
  ! The loop is directly attached to the `acc kernels` directive, so it is not
  ! wrapped in an scf.execute_region either.
  ! CHECK2: {{.*}}do_loop_with_cycle_goto.f90":[[@LINE+2]]:3){{.*}}not yet implemented: unstructured do loop in acc kernels
  !$acc kernels
  do i = 1, n
    if (i == 3) cycle
    a(i) = b(i) + 1.0
    if (i == 7) goto 200
    a(i) = a(i) * 2.0
  end do
200 continue
  !$acc end kernels


end subroutine

//--- nested_goto_loop.f90

subroutine nested_goto_loop()
  integer :: i, j
  integer, parameter :: n = 10, m = 5
  real, dimension(n,m) :: a, b

  ! Nested loop with goto from inner to outer - should NOT convert to acc.loop
  ! CHECK3: {{.*}}nested_goto_loop.f90":[[@LINE+2]]:3){{.*}}not yet implemented: unstructured do loop in acc kernels
  !$acc kernels
  do i = 1, n
    do j = 1, m
      a(i,j) = b(i,j) + 1.0
      if (i * j > 20) goto 300  ! Exit both loops
    end do
  end do
300 continue
  !$acc end kernels


end subroutine

//--- collapse_lt.f90

! collapse(2) over a 3-control do concurrent: collapse < control count (N < C).
subroutine combined(i, j, k)
  integer :: i, j, k
  integer :: a(i,j,k)
  ! CHECK7: {{.*}}collapse_lt.f90":[[@LINE+1]]:9){{.*}}not yet implemented: OpenACC LOOP COLLAPSE less than DO CONCURRENT control count
  !$acc parallel loop collapse(2)
  do concurrent (i=1:10, j=1:100, k=1:200)
    a(i,j,k) = a(i,j,k) + 1
  end do
end subroutine

//--- collapse_gt.f90

! collapse(3) over a 2-control do concurrent + inner regular do (N > C, mixed).
! Semantics accepts it: the inner do supplies the extra level.
subroutine combined(i, j, k)
  integer :: i, j, k
  integer :: a(i,j,k)
  ! CHECK8: {{.*}}collapse_gt.f90":[[@LINE+1]]:9){{.*}}not yet implemented: OpenACC LOOP COLLAPSE greater than DO CONCURRENT control count
  !$acc parallel loop collapse(3)
  do concurrent (i=1:10, j=1:100)
    do k = 1, 200
      a(i,j,k) = a(i,j,k) + 1
    end do
  end do
end subroutine

//--- collapse_nested.f90

! !$acc parallel loop collapse(N) over a nested do concurrent.
subroutine combined(i, j, k)
  integer :: i, j, k
  integer :: a(i,j,k)
  !$acc parallel loop collapse(3)
  do i = 1, 10
    ! CHECK6: {{.*}}collapse_nested.f90":[[@LINE+1]]:5){{.*}}not yet implemented: OpenACC LOOP with nested DO CONCURRENT
    do concurrent (j=1:100, k=1:200)
      a(i,j,k) = a(i,j,k) + 1
    end do
  end do

end subroutine

//--- plain_do_in_seq_loop.f90

! A plain DO in the body of a seq loop, leaving the seq loop with a named EXIT.
! The seq acc.loop does not own the plain DO, so it does not privatize its
! induction variable.
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
      ! The `!` DO over k lowers into the seq acc.loop, which privatizes k.
      ! Lowering stops at the `!` DO over l: the acc.loop does not privatize l.
      ! CHECK10: {{.*}}plain_do_in_seq_loop.f90":[[@LINE+1]]:7){{.*}}not yet implemented: unstructured do loop in acc kernels
      do l = 1, p
        if (b(l, k, i) > 0.0) then
          kref(i) = k
          exit kloop
        end if
      end do
    end do kloop
  end do
  !$acc end kernels

! PFT10-LABEL: Subroutine plain_do_in_seq_loop
! PFT10:         <<OpenACCConstruct~>>
! PFT10-NEXT:      <<DoConstruct~>>
! PFT10-NEXT:        NonLabelDoStmt {{.*}}: do i = 1, n
! PFT10:             <<OpenACCConstruct!>>
! PFT10-NEXT:          <<DoConstruct!>>
! PFT10-NEXT:            NonLabelDoStmt {{.*}}: kloop: do k = 1, m
! PFT10-NEXT:            <<DoConstruct!>>
! PFT10-NEXT:              NonLabelDoStmt {{.*}}: do l = 1, p
! PFT10:                   ExitStmt! {{.*}}: exit kloop
! PFT10:                 <<End DoConstruct!>>
! PFT10:               <<End DoConstruct!>>
! PFT10:             <<End OpenACCConstruct!>>
! PFT10:           <<End DoConstruct~>>
! PFT10:         <<End OpenACCConstruct~>>

end subroutine

//--- collapse_force_sibling_do.f90

! A seq loop with collapse(force:2). The collapse absorbs only the first DO
! nested in the outer one, so the seq acc.loop privatizes k and l but not q,
! although the DO over q is as deeply nested as the one over l.
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
      ! The `!` DO over k lowers into the seq acc.loop, which privatizes k and l.
      ! The structured DO over l lowers as a structured loop. Lowering stops at
      ! the `!` DO over q: the acc.loop does not privatize q.
      ! CHECK11: {{.*}}collapse_force_sibling_do.f90":[[@LINE+1]]:7){{.*}}not yet implemented: unstructured do loop in acc kernels
      do q = 1, p
        if (b(q, k, i) > 0.0) then
          kref(i) = k
          exit kloop
        end if
      end do
    end do kloop
  end do
  !$acc end kernels

! PFT11-LABEL: Subroutine collapse_force_sibling_do
! PFT11:         <<OpenACCConstruct~>>
! PFT11-NEXT:      <<DoConstruct~>>
! PFT11-NEXT:        NonLabelDoStmt {{.*}}: do i = 1, n
! PFT11:             <<OpenACCConstruct!>>
! PFT11-NEXT:          <<DoConstruct!>>
! PFT11-NEXT:            NonLabelDoStmt {{.*}}: kloop: do k = 1, m
! PFT11-NEXT:            <<DoConstruct>>
! PFT11-NEXT:              NonLabelDoStmt {{.*}}: do l = 1, p
! PFT11:                 <<End DoConstruct>>
! PFT11-NEXT:            <<DoConstruct!>>
! PFT11-NEXT:              NonLabelDoStmt {{.*}}: do q = 1, p
! PFT11:                   ExitStmt! {{.*}}: exit kloop
! PFT11:                 <<End DoConstruct!>>
! PFT11:               <<End DoConstruct!>>
! PFT11:             <<End OpenACCConstruct!>>
! PFT11:           <<End DoConstruct~>>
! PFT11:         <<End OpenACCConstruct~>>

end subroutine

//--- collapse_force_sibling_same_var.f90

! Like collapse_force_sibling_do, but the sibling DO reuses the variable l of
! the collapsed DO. The seq acc.loop privatizes l, yet it does not own the
! sibling DO, so the sibling DO stays TODO as the one over q does: ownership
! follows the DOs of the collapse, not the variables they privatize.
subroutine collapse_force_sibling_same_var(n, m, p, b, kref)
  integer :: n, m, p
  real :: b(p, m, n)
  integer :: kref(n)
  integer :: i, k, l
  !$acc kernels
  do i = 1, n
    kref(i) = 0
    !$acc loop seq collapse(force:2)
    kloop: do k = 1, m
      do l = 1, p
        kref(i) = kref(i) + 1
      end do
      ! The `!` DO over k lowers into the seq acc.loop, which privatizes k and l.
      ! The structured DO over l lowers as a structured loop. Lowering stops at
      ! the second, `!` DO over l: the seq acc.loop does not own it.
      ! CHECK19: {{.*}}collapse_force_sibling_same_var.f90":[[@LINE+1]]:7){{.*}}not yet implemented: unstructured do loop in acc kernels
      do l = 1, p
        if (b(l, k, i) > 0.0) then
          kref(i) = k
          exit kloop
        end if
      end do
    end do kloop
  end do
  !$acc end kernels

! PFT19-LABEL: Subroutine collapse_force_sibling_same_var
! PFT19:         <<OpenACCConstruct~>>
! PFT19-NEXT:      <<DoConstruct~>>
! PFT19-NEXT:        NonLabelDoStmt {{.*}}: do i = 1, n
! PFT19:             <<OpenACCConstruct!>>
! PFT19-NEXT:          <<DoConstruct!>>
! PFT19-NEXT:            NonLabelDoStmt {{.*}}: kloop: do k = 1, m
! PFT19-NEXT:            <<DoConstruct>>
! PFT19-NEXT:              NonLabelDoStmt {{.*}}: do l = 1, p
! PFT19:                 <<End DoConstruct>>
! PFT19-NEXT:            <<DoConstruct!>>
! PFT19-NEXT:              NonLabelDoStmt {{.*}}: do l = 1, p
! PFT19:                   ExitStmt! {{.*}}: exit kloop
! PFT19:                 <<End DoConstruct!>>
! PFT19:               <<End DoConstruct!>>
! PFT19:             <<End OpenACCConstruct!>>
! PFT19:           <<End DoConstruct~>>
! PFT19:         <<End OpenACCConstruct~>>

end subroutine

//--- seq_exit_device_type_seq.f90

! A loop with an early EXIT in the kernels region that is seq only for one
! device_type. For the default device_type it is not seq.
subroutine seq_exit_device_type_seq(m, b, kfirst)
  integer :: m, kfirst
  real :: b(m)
  integer :: k
  !$acc kernels
  kfirst = 0
  ! Lowering stops at the `!` DO over k: its acc.loop is not seq for the
  ! default device_type.
  ! CHECK16: {{.*}}seq_exit_device_type_seq.f90":[[@LINE+2]]:3){{.*}}not yet implemented: unstructured do loop in acc kernels
  !$acc loop device_type(nvidia) seq
  do k = 1, m
    if (b(k) > 0.0) then
      kfirst = k
      exit
    end if
  end do
  !$acc end kernels

! PFT16-LABEL: Subroutine seq_exit_device_type_seq
! PFT16:         <<OpenACCConstruct!>>
! PFT16:           <<OpenACCConstruct!>>
! PFT16:             <<DoConstruct!>>
! PFT16:               NonLabelDoStmt {{.*}}: do k = 1, m
! PFT16:                 ExitStmt! {{.*}}: exit
! PFT16:             <<End DoConstruct!>>
! PFT16:           <<End OpenACCConstruct!>>
! PFT16:         <<End OpenACCConstruct!>>


end subroutine

//--- auto_exit_in_kernels.f90

! An `acc loop auto` with an early EXIT in the kernels region.
subroutine auto_exit_in_kernels(m, b, kfirst)
  integer :: m, kfirst
  real :: b(m)
  integer :: k
  !$acc kernels
  kfirst = 0
  ! Lowering stops at the `!` DO over k: its acc.loop is auto, not seq.
  ! CHECK17: {{.*}}auto_exit_in_kernels.f90":[[@LINE+2]]:3){{.*}}not yet implemented: unstructured do loop in acc kernels
  !$acc loop auto
  do k = 1, m
    if (b(k) > 0.0) then
      kfirst = k
      exit
    end if
  end do
  !$acc end kernels

! PFT17-LABEL: Subroutine auto_exit_in_kernels
! PFT17:         <<OpenACCConstruct!>>
! PFT17:           <<OpenACCConstruct!>>
! PFT17:             <<DoConstruct!>>
! PFT17:               NonLabelDoStmt {{.*}}: do k = 1, m
! PFT17:                 ExitStmt! {{.*}}: exit
! PFT17:             <<End DoConstruct!>>
! PFT17:           <<End OpenACCConstruct!>>
! PFT17:         <<End OpenACCConstruct!>>


end subroutine

//--- loop_exit_in_kernels.f90

! An `acc loop` without a parallelism clause, with an early EXIT, in the
! kernels region. In kernels it is treated as auto.
subroutine loop_exit_in_kernels(m, b, kfirst)
  integer :: m, kfirst
  real :: b(m)
  integer :: k
  !$acc kernels
  kfirst = 0
  ! Lowering stops at the `!` DO over k: its acc.loop is auto, not seq.
  ! CHECK18: {{.*}}loop_exit_in_kernels.f90":[[@LINE+2]]:3){{.*}}not yet implemented: unstructured do loop in acc kernels
  !$acc loop
  do k = 1, m
    if (b(k) > 0.0) then
      kfirst = k
      exit
    end if
  end do
  !$acc end kernels

! PFT18-LABEL: Subroutine loop_exit_in_kernels
! PFT18:         <<OpenACCConstruct!>>
! PFT18:           <<OpenACCConstruct!>>
! PFT18:             <<DoConstruct!>>
! PFT18:               NonLabelDoStmt {{.*}}: do k = 1, m
! PFT18:                 ExitStmt! {{.*}}: exit
! PFT18:             <<End DoConstruct!>>
! PFT18:           <<End OpenACCConstruct!>>
! PFT18:         <<End OpenACCConstruct!>>


end subroutine
