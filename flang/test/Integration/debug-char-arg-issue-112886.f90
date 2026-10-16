! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | \
! RUN:   FileCheck %s --check-prefix=LLVM

! Test CHARACTER argument
subroutine char_arg(str1)
  character(len=5) :: str1
  print *, str1
end subroutine

! Test CHARACTER argument with different length
subroutine char_arg_len10(str2)
  character(len=10) :: str2
  print *, str2
end subroutine

! Test multiple CHARACTER arguments
subroutine multi_char_args(s1, s2, s3)
  character(len=5) :: s1
  character(len=8) :: s2
  character(len=3) :: s3
  print *, s1, s2, s3
end subroutine

! Test mixed argument types (CHARACTER and INTEGER)
subroutine mixed_args(n, str, m)
  integer :: n
  character(len=7) :: str
  integer :: m
  print *, n, str, m
end subroutine

program test
  call char_arg('hello')
  call char_arg_len10('hello test')
  call multi_char_args('abc', 'test123', 'xyz')
  call mixed_args(1, 'fortran', 2)
end program test

! LLVM-DAG: !DILocalVariable(name: "str1", arg: 1
! LLVM-DAG: !DILocalVariable(name: "str2", arg: 1
! LLVM-DAG: !DILocalVariable(name: "s1", arg: 1
! LLVM-DAG: !DILocalVariable(name: "s2", arg: 2
! LLVM-DAG: !DILocalVariable(name: "s3", arg: 3
! LLVM-DAG: !DILocalVariable(name: "n", arg: 1
! LLVM-DAG: !DILocalVariable(name: "str", arg: 2
! LLVM-DAG: !DILocalVariable(name: "m", arg: 3
