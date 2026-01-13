! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
module equiv_kind_m
  implicit none
  integer, parameter :: knd = kind(42)
  integer, parameter :: dim_2 = 1_knd
  integer, parameter :: n = 3_knd
  integer, parameter :: i_start = 1_knd
contains
subroutine test()
  integer(knd) :: a(n),b(n,n)
  character(len=5) :: small_ch
  character(len=20) :: large_ch

  equivalence (a(1_knd),b(1_knd,dim_2))
  !CHECK: EQUIVALENCE (a(1_4), b(1_4,1_4))
  equivalence (small_ch, large_ch(i_start:5_knd))
  !CHECK: EQUIVALENCE (small_ch, large_ch(1_4:5_4))
end subroutine test
end module equiv_kind_m
