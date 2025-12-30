! Test -mlink-builtin-bitcode flag
! RUN: %flang -emit-llvm -c -o %t.bc %S/Inputs/libfun.f90
! RUN: %flang_fc1 -emit-llvm -o - -mlink-builtin-bitcode %t.bc %s 2>&1 | FileCheck %s

! CHECK: define internal void @libfun_

! RUN: not %flang_fc1 -emit-llvm -o - -mlink-builtin-bitcode %no-%t.bc %s 2>&1 | FileCheck %s --check-prefix=ERROR

! ERROR: error: could not open {{.*}}.bc

external libfun
parameter(i=1)
integer :: j
call libfun(j)
end program
