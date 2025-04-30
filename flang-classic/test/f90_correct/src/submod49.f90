! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

MODULE m
IMPLICIT NONE

  INTERFACE
    module function F1(A) result(B)
    integer, dimension(10), intent(in) :: A
    integer, allocatable :: B(:)
    end function 

    module function F2(C) result(D)
    integer, dimension(10), intent(in) :: C
    integer, allocatable :: D(:)
    end function 
  END INTERFACE
END MODULE

SUBMODULE (m) n
  CONTAINS
    MODULE FUNCTION F1(A) result(B) !{error "PGF90-S-1061-The definition of function return type of b does not match its declaration type"}
      integer, dimension(10), intent(in) :: A
      real, allocatable :: B(:)
    end function

    MODULE FUNCTION F2(C) result(D) !{error "PGF90-S-1061-The definition of function return type of d does not match its declaration type"}
      integer, dimension(10), intent(in) :: C
      integer, dimension(10) :: D
    end function
END SUBMODULE

