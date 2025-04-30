! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

MODULE m
  INTERFACE
  INTEGER MODULE FUNCTION F1(arg1)
    INTEGER, intent(inout) :: arg1
  END FUNCTION
  
  INTEGER MODULE FUNCTION F2(arg2)
    INTEGER, intent(inout) :: arg2
  END FUNCTION

  END INTERFACE
END MODULE

SUBMODULE (m) n
  CONTAINS
    REAL MODULE FUNCTION F1(arg1) !{error "PGF90-S-1061-The definition of function return type of f1 does not match its declaration type"}
      INTEGER, intent(inout) :: arg1
    END FUNCTION F1

    REAL MODULE FUNCTION F2(arg2) RESULT(ret) !{error "PGF90-S-1061-The definition of function return type of f2 does not match its declaration type"}
      INTEGER, intent(inout) :: arg2
      ret = arg2
    END FUNCTION

END SUBMODULE

