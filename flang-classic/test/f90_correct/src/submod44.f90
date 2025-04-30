! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

MODULE m
  INTEGER :: res
  INTERFACE
    MODULE SUBROUTINE sub1(arg1)
      INTEGER, intent(inout) :: arg1
    END SUBROUTINE

    INTEGER MODULE FUNCTION F1(ARG4)
      INTEGER, INTENT(INOUT) :: ARG4
    END FUNCTION

    INTEGER MODULE FUNCTION F2(ARG5)
      INTEGER, INTENT(INOUT) :: ARG5
    END FUNCTION
  END INTERFACE
END MODULE

SUBMODULE (m) n
INTERFACE
  MODULE SUBROUTINE SUB2(ARG3, ARG4)
    INTEGER, INTENT(INOUT) :: ARG3, ARG4
  END SUBROUTINE

END INTERFACE
  CONTAINS
    INTEGER MODULE FUNCTION F1(ARG4, ARG5) !{error "PGF90-S-1059-The definition of subprogram f1 does not have the same number of arguments as its declaration"}
      INTEGER, INTENT(INOUT) :: ARG4, ARG5
    END FUNCTION
END SUBMODULE

SUBMODULE (m:n) k
  CONTAINS
    MODULE SUBROUTINE sub1(arg1, arg2) !{error "PGF90-S-1059-The definition of subprogram sub1 does not have the same number of arguments as its declaration"}
      INTEGER, intent(inout) :: arg1
      INTEGER, intent(inout) :: arg2
      arg1 = arg1 + 1
    END SUBROUTINE sub1

    MODULE SUBROUTINE SUB2(arg3) !{error "PGF90-S-1059-The definition of subprogram sub2 does not have the same number of arguments as its declaration"}
      INTEGER, intent(inout) :: arg3
    END SUBROUTINE SUB2

    INTEGER MODULE FUNCTION F2(ARG5, ARG6) !{error "PGF90-S-1059-The definition of subprogram f2 does not have the same number of arguments as its declaration"}
      INTEGER, INTENT(INOUT) :: ARG5, ARG6
    END FUNCTION
END SUBMODULE

