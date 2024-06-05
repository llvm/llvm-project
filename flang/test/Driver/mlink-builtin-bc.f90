
!----------
! RUN lines
!----------
! Embed something that can be easily checked
! RUN: %flang_fc1 -emit-llvm -triple x86_64-unknown-linux-gnu -o - -mlink-builtin-bitcode %S/Inputs/bclib.bc %s 2>&1 | FileCheck %s

! CHECK: define internal void @libfun_

! RUN1: not %flang_fc1 -emit-llvm -triple x86_64-unknown-linux-gnu -o - -mlink-builtin-bitcode %S/Inputs/no-bclib.bc %s 2>&1 | FileCheck %s

! ERROR1: error: could not open {{.*}} no-bclib.bc

external libfun
parameter(i=1)
integer :: j
call libfun(j)
end program
