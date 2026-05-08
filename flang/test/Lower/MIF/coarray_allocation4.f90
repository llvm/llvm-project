! RUN: not %flang_fc1 -emit-hlfir -fcoarray %s -o - 2>&1 | FileCheck %s

!CHECK: not yet implemented: non-ALLOCATABLE SAVE Coarray outside the main program.

subroutine test_coarray_save()
    implicit none
    real, SAVE :: n[*]
end subroutine test_coarray_save
