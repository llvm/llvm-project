! Tests for -fprofile-generate and -fprofile-use flag compatibility. These two
! flags behave similarly to their GCC counterparts:
!
! -fprofile-generate         Generates the profile file ./default.profraw
! -fprofile-use=<dir>/file   Uses the profile file <dir>/file

! On AIX, -flto used to be required with -fprofile-generate. gcc-flag-compatibility-aix.c is used to do the testing on AIX with -flto
! RUN: %flang %s -c -S -o - -emit-llvm -fprofile-generate | FileCheck -check-prefix=PROFILE-GEN %s
! PROFILE-GEN: @__profc_{{_?}}main = {{(private|internal)}} global [1 x i64] zeroinitializer, section
! PROFILE-GEN: @__profd_{{_?}}main =

! Check that -fprofile-use=some/path/file.prof reads some/path/file.prof
! This uses LLVM IR format profile.
! RUN: rm -rf %t.dir
! RUN: mkdir -p %t.dir/some/path
! RUN: llvm-profdata merge %S/Inputs/gcc-flag-compatibility_IR.proftext -o %t.dir/some/path/file.prof
! RUN: %flang %s -o - -emit-llvm -S -fprofile-use=%t.dir/some/path/file.prof | FileCheck -check-prefix=PROFILE-USE-IR1 %s
! RUN: llvm-profdata merge %S/Inputs/gcc-flag-compatibility_IR_entry.proftext -o %t.dir/some/path/file.prof
! RUN: %flang %s -o - -emit-llvm -S -fprofile-use=%t.dir/some/path/file.prof | FileCheck -check-prefix=PROFILE-USE-IR2 %s
! PROFILE-USE-IR1: = !{!"branch_weights", i32 100, i32 1}
! PROFILE-USE-IR2: = !{!"branch_weights", i32 1, i32 100}

program main
  implicit none
  integer :: i
  integer :: X = 0

  do i = 0, 99
     X = X + i
  end do

end program main
