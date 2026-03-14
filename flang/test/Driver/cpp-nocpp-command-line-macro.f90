!-----------
! RUN lines
!-----------
! RUN: %flang_fc1 -E %s 2>&1 | FileCheck %s --check-prefix=UNDEFINED
! RUN: %flang_fc1 -E -cpp -DX=A %s 2>&1 | FileCheck %s --check-prefix=DEFINED
! RUN: %flang_fc1 -E -nocpp -DX=A %s 2>&1 | FileCheck %s --check-prefix=UNDEFINED

! UNDEFINED:program B
! UNDEFINED-NOT:program A

! DEFINED:program A
! DEFINED-NOT:program B

#ifdef X
program X
#else
program B
#endif
end
