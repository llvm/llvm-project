! RUN: split-file %s %t
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/do_loop_with_cycle_goto.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK2
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/nested_goto_loop.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK3
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/collapse_lt.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK7
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/collapse_gt.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK8
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/collapse_nested.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK6
! RUN: bbc -fopenacc -pft-test -o /dev/null %t/seq_exit_in_kernels.f90 | FileCheck %s --check-prefix=PFT9
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/seq_exit_in_kernels.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK9

//--- do_loop_with_cycle_goto.f90

subroutine do_loop_with_cycle_goto()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  ! Do loop with cycle and goto - unstructured control flow is not converted.
  ! The loop is directly attached to the `acc kernels` directive, so it is not
  ! wrapped in an scf.execute_region either.
  !$acc kernels
  do i = 1, n
    if (i == 3) cycle
    a(i) = b(i) + 1.0
    if (i == 7) goto 200
    a(i) = a(i) * 2.0
  end do
200 continue
  !$acc end kernels

! CHECK2: not yet implemented: unstructured do loop in acc kernels

end subroutine

//--- nested_goto_loop.f90

subroutine nested_goto_loop()
  integer :: i, j
  integer, parameter :: n = 10, m = 5
  real, dimension(n,m) :: a, b

  ! Nested loop with goto from inner to outer - should NOT convert to acc.loop
  !$acc kernels
  do i = 1, n
    do j = 1, m
      a(i,j) = b(i,j) + 1.0
      if (i * j > 20) goto 300  ! Exit both loops
    end do
  end do
300 continue
  !$acc end kernels

! CHECK3: not yet implemented: unstructured do loop in acc kernels

end subroutine

//--- collapse_lt.f90

! collapse(2) over a 3-control do concurrent: collapse < control count (N < C).
subroutine combined(i, j, k)
  integer :: i, j, k
  integer :: a(i,j,k)
  !$acc parallel loop collapse(2)
  do concurrent (i=1:10, j=1:100, k=1:200)
    a(i,j,k) = a(i,j,k) + 1
  end do
  ! CHECK7: not yet implemented: OpenACC LOOP COLLAPSE less than DO CONCURRENT control count
end subroutine

//--- collapse_gt.f90

! collapse(3) over a 2-control do concurrent + inner regular do (N > C, mixed).
! Semantics accepts it: the inner do supplies the extra level.
subroutine combined(i, j, k)
  integer :: i, j, k
  integer :: a(i,j,k)
  !$acc parallel loop collapse(3)
  do concurrent (i=1:10, j=1:100)
    do k = 1, 200
      a(i,j,k) = a(i,j,k) + 1
    end do
  end do
  ! CHECK8: not yet implemented: OpenACC LOOP COLLAPSE greater than DO CONCURRENT control count
end subroutine

//--- collapse_nested.f90

! !$acc parallel loop collapse(N) over a nested do concurrent.
subroutine combined(i, j, k)
  integer :: i, j, k
  integer :: a(i,j,k)
  !$acc parallel loop collapse(3)
  do i = 1, 10
    do concurrent (j=1:100, k=1:200)
      a(i,j,k) = a(i,j,k) + 1
    end do
  end do
  ! CHECK6: not yet implemented: OpenACC LOOP with nested DO CONCURRENT

end subroutine

//--- seq_exit_in_kernels.f90

! A seq loop with an early EXIT directly in the `acc kernels` region: no
! enclosing construct contains its control flow, so the kernels construct
! itself stays unstructured.
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

! PFT9-LABEL: Subroutine seq_exit_in_kernels
! PFT9:         <<OpenACCConstruct!>>
! PFT9:           <<OpenACCConstruct!>>
! PFT9-NEXT:        <<DoConstruct!>>
! PFT9-NEXT:          NonLabelDoStmt {{.*}}: do k = 1, m
! PFT9:               ExitStmt! {{.*}}: exit
! PFT9:             <<End DoConstruct!>>
! PFT9:           <<End OpenACCConstruct!>>
! PFT9:         <<End OpenACCConstruct!>>

! Lowering stops at the `!` DO over k (line 13): its `acc loop` construct sits
! directly in the `!` kernels construct, so nothing contains its control flow.
! CHECK9: {{.*}}seq_exit_in_kernels.f90":13:3){{.*}}not yet implemented: unstructured do loop in acc kernels

end subroutine
