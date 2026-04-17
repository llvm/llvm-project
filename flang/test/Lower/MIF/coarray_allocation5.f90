! RUN: not %flang_fc1 -emit-hlfir -fcoarray %s -o - 2>&1 | FileCheck %s

!CHECK: not yet implemented: non-ALLOCATABLE SAVE Coarray outside the main program.

module m_coarray_test
    implicit none
    real :: module_coarray[*]
end module m_coarray_test
