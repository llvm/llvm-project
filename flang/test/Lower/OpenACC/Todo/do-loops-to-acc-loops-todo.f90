! RUN: split-file --leading-lines %s %t
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/do_loop_with_cycle_goto.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK2
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/nested_goto_loop.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK3
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/collapse_lt.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK7
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/collapse_gt.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK8
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/collapse_nested.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK6
! RUN: bbc -fopenacc -pft-test -o /dev/null %t/seq_exit_in_if.f90 | FileCheck %s --check-prefix=PFT12
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/seq_exit_in_if.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK12
! RUN: bbc -fopenacc -pft-test -o /dev/null %t/seq_exit_in_select_case.f90 | FileCheck %s --check-prefix=PFT13
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/seq_exit_in_select_case.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK13
! RUN: bbc -fopenacc -pft-test -o /dev/null %t/seq_exit_in_block.f90 | FileCheck %s --check-prefix=PFT14
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/seq_exit_in_block.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK14
! RUN: bbc -fopenacc -pft-test -o /dev/null %t/seq_exit_in_do_while.f90 | FileCheck %s --check-prefix=PFT15
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/seq_exit_in_do_while.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK15
! RUN: bbc -fopenacc -pft-test -o /dev/null %t/seq_exit_device_type_seq.f90 | FileCheck %s --check-prefix=PFT16
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/seq_exit_device_type_seq.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK16
! RUN: bbc -fopenacc -pft-test -o /dev/null %t/auto_exit_in_kernels.f90 | FileCheck %s --check-prefix=PFT17
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/auto_exit_in_kernels.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK17
! RUN: bbc -fopenacc -pft-test -o /dev/null %t/loop_exit_in_kernels.f90 | FileCheck %s --check-prefix=PFT18
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/loop_exit_in_kernels.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK18

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

//--- seq_exit_in_if.f90

! A seq loop with an early EXIT inside an IF in the kernels region. The IF
! holding the `acc loop` construct stays unstructured, so the seq loop is not
! contained.
subroutine seq_exit_in_if(m, b, kfirst, c)
  integer :: m, kfirst
  real :: b(m)
  logical :: c
  integer :: k
  !$acc kernels
  kfirst = 0
  if (c) then
    ! Lowering stops at the seq loop's `!` DO over k. The IF around the seq
    ! loop's `acc loop` construct is unstructured and is not the kernels
    ! construct.
    ! CHECK12: {{.*}}seq_exit_in_if.f90":[[@LINE+2]]:5){{.*}}not yet implemented: unstructured do loop in acc kernels
    !$acc loop seq
    do k = 1, m
      if (b(k) > 0.0) then
        kfirst = k
        exit
      end if
    end do
  end if
  !$acc end kernels

! PFT12-LABEL: Subroutine seq_exit_in_if
! PFT12:         <<OpenACCConstruct>>
! PFT12:           <<IfConstruct!>>
! PFT12:             <<^OpenACCConstruct!>>
! PFT12:               <<DoConstruct!>>
! PFT12:                 NonLabelDoStmt {{.*}}: do k = 1, m
! PFT12:                   ExitStmt! {{.*}}: exit
! PFT12:               <<End DoConstruct!>>
! PFT12:             <<End OpenACCConstruct!>>
! PFT12:           <<End IfConstruct!>>
! PFT12:         <<End OpenACCConstruct>>


end subroutine

//--- seq_exit_in_select_case.f90

! A seq loop with an early EXIT inside a SELECT CASE in the kernels region.
subroutine seq_exit_in_select_case(m, b, kfirst, s)
  integer :: m, kfirst
  real :: b(m)
  integer :: s
  integer :: k
  !$acc kernels
  kfirst = 0
  select case (s)
  case (1)
    ! Lowering stops at the seq loop's `!` DO over k. The SELECT CASE around the
    ! seq loop's `acc loop` construct is unstructured and is not the kernels
    ! construct.
    ! CHECK13: {{.*}}seq_exit_in_select_case.f90":[[@LINE+2]]:5){{.*}}not yet implemented: unstructured do loop in acc kernels
    !$acc loop seq
    do k = 1, m
      if (b(k) > 0.0) then
        kfirst = k
        exit
      end if
    end do
  end select
  !$acc end kernels

! PFT13-LABEL: Subroutine seq_exit_in_select_case
! PFT13:         <<OpenACCConstruct!>>
! PFT13:           <<CaseConstruct!>>
! PFT13:             <<OpenACCConstruct!>>
! PFT13:               <<DoConstruct!>>
! PFT13:                 NonLabelDoStmt {{.*}}: do k = 1, m
! PFT13:                   ExitStmt! {{.*}}: exit
! PFT13:               <<End DoConstruct!>>
! PFT13:             <<End OpenACCConstruct!>>
! PFT13:           <<End CaseConstruct!>>
! PFT13:         <<End OpenACCConstruct!>>


end subroutine

//--- seq_exit_in_block.f90

! A seq loop with an early EXIT inside a BLOCK construct in the kernels
! region.
subroutine seq_exit_in_block(m, b, kfirst)
  integer :: m, kfirst
  real :: b(m)
  integer :: k
  !$acc kernels
  kfirst = 0
  block
    ! Lowering stops at the seq loop's `!` DO over k. The BLOCK around the seq
    ! loop's `acc loop` construct is unstructured and is not the kernels
    ! construct.
    ! CHECK14: {{.*}}seq_exit_in_block.f90":[[@LINE+2]]:5){{.*}}not yet implemented: unstructured do loop in acc kernels
    !$acc loop seq
    do k = 1, m
      if (b(k) > 0.0) then
        kfirst = k
        exit
      end if
    end do
  end block
  !$acc end kernels

! PFT14-LABEL: Subroutine seq_exit_in_block
! PFT14:         <<OpenACCConstruct!>>
! PFT14:           <<BlockConstruct!>>
! PFT14:             <<OpenACCConstruct!>>
! PFT14:               <<DoConstruct!>>
! PFT14:                 NonLabelDoStmt {{.*}}: do k = 1, m
! PFT14:                   ExitStmt! {{.*}}: exit
! PFT14:               <<End DoConstruct!>>
! PFT14:             <<End OpenACCConstruct!>>
! PFT14:           <<End BlockConstruct!>>
! PFT14:         <<End OpenACCConstruct!>>


end subroutine

//--- seq_exit_in_do_while.f90

! A seq loop with an early EXIT inside a DO WHILE in the kernels region. A DO
! WHILE is never reclassified as structured with unstructured internals.
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
    ! Lowering stops at the seq loop's `!` DO over k. The DO WHILE around the
    ! seq loop's `acc loop` construct is unstructured and is not the kernels
    ! construct.
    ! CHECK15: {{.*}}seq_exit_in_do_while.f90":[[@LINE+2]]:5){{.*}}not yet implemented: unstructured do loop in acc kernels
    !$acc loop seq
    do k = 1, m
      if (b(k) > 0.0) then
        kfirst = k
        exit
      end if
    end do
  end do
  !$acc end kernels

! PFT15-LABEL: Subroutine seq_exit_in_do_while
! PFT15:         <<OpenACCConstruct!>>
! PFT15:           <<DoConstruct!>>
! PFT15:             NonLabelDoStmt! {{.*}}: do while(i < n)
! PFT15:             <<OpenACCConstruct!>>
! PFT15:               <<DoConstruct!>>
! PFT15:                 NonLabelDoStmt {{.*}}: do k = 1, m
! PFT15:                   ExitStmt! {{.*}}: exit
! PFT15:               <<End DoConstruct!>>
! PFT15:             <<End OpenACCConstruct!>>
! PFT15:           <<End DoConstruct!>>
! PFT15:         <<End OpenACCConstruct!>>


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
