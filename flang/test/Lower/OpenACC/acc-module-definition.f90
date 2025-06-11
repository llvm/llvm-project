! RUN: rm -fr %t && mkdir -p %t && cd %t
! RUN: bbc -fopenacc -emit-fir %s
! RUN: cat mod1.mod | FileCheck %s

!CHECK-LABEL: module mod1
module mod1
    contains
      !CHECK subroutine callee(aa)
      subroutine callee(aa)
      !CHECK: !$acc routine seq
      !$acc routine seq
        integer :: aa
        aa = 1
      end subroutine
      !CHECK: end
      !CHECK: end
end module