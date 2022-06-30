
!----------
! RUN lines
!----------
! RUN: %flang_fc1 -fdebug-dump-all %s 2>&1 | FileCheck %s

! CHECK: Flang: parse tree dump
! CHECK: Flang: symbols dump

parameter(i=1)
integer :: j
end program
