! Each sub-file exercises a different unstructured-CFG pattern inside a
! combined `acc parallel loop` construct (default parallelism is
! `independent`).

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s --check-prefix=CYCLE2-OK
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir --emit-independent-loops-as-unstructured=false %s -o - 2>&1 | FileCheck %s --check-prefix=CYCLE2

subroutine test_unstructured_collapse_cycle(a)
  integer :: i, j, jdiag
  real(8) :: a(:,:)
  jdiag = 4
  !$acc parallel loop collapse(2) copy(a)
  do j = 1, 8
    do i = 1, 8
      if (i == jdiag) then
        a(i, j) = 0.0d0
        cycle
      end if
      a(i, j) = real(i + j, 8)
    end do
  end do
  !$acc end parallel loop
end subroutine

! CYCLE2: not yet implemented: unstructured do loop in combined acc construct

! CYCLE2-OK-LABEL: func.func @_QPtest_unstructured_collapse_cycle
! CYCLE2-OK: acc.parallel combined(loop)
! CYCLE2-OK: acc.loop combined(parallel)
