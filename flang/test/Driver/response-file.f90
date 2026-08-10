! Test that the driver can process response files.

! RUN: echo "-DTEST" > %basename_t.rsp
! RUN: %flang -E -cpp @%basename_t.rsp %s -o - | FileCheck %s
! RUN: %flang_fc1 -E -cpp @%basename_t.rsp %s -o - | FileCheck %s
! RUN: not %flang %basename_t.rsp %s -o /dev/null
! RUN: not %flang_fc1 %basenamt_t.rsp %s -o /dev/null

! CHECK-LABEL: program test
! CHECK: end program

#ifdef TEST
program test
end program
#else
We should have read the define from the response file.
#endif
