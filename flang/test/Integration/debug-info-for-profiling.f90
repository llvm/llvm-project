! Test to check the option "-fdebug-info-for-profiling".

! RUN: %flang -S -emit-llvm -fdebug-info-for-profiling -g -o - %s | FileCheck %s

! CHECK: !DICompileUnit({{.*}}debugInfoForProfiling: true{{.*}})
! CHECK: !DILexicalBlockFile(
! CHECK: discriminator:

program test
   implicit none
   integer :: i, sum
   sum = 0
   do i = 1, 20
      if (mod(i, 2) == 0) then
         sum = sum + compute(i)
      else
         sum = sum + compute(i)*2
      end if
   end do

contains
   integer function compute(x)
      implicit none
      integer, intent(in) :: x
      if (x < 10) then
         compute = x*x
      else
         compute = x + 5
      end if
   end function compute

end program test
