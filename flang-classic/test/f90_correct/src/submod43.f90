! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

MODULE m
  INTEGER :: res
  INTERFACE
    MODULE SUBROUTINE sub1(arg1)
      INTEGER, intent(inout) :: arg1
    END SUBROUTINE

    MODULE SUBROUTINE sub2(arg2)
      INTEGER, intent(inout) :: arg2
    END SUBROUTINE

    INTEGER MODULE FUNCTION F1(ARG3)
      INTEGER, INTENT(INOUT) :: ARG3
    END FUNCTION

    INTEGER MODULE FUNCTION F2(ARG4)
      INTEGER, INTENT(INOUT) :: ARG4
    END FUNCTION
  END INTERFACE
END MODULE

SUBMODULE (m) n
INTERFACE
  MODULE SUBROUTINE sub4(arg4)
    INTEGER, intent(inout) :: arg4
  END SUBROUTINE

END INTERFACE
  CONTAINS
    MODULE SUBROUTINE sub1(arg2) !{error "PGF90-S-1057-Definition argument name arg2 does not match declaration argument name arg1"}
      INTEGER, intent(inout) :: arg2
      print *, arg2
    END SUBROUTINE sub1

    INTEGER MODULE FUNCTION F1(ARG4) !{error "PGF90-S-1057-Definition argument name arg4 does not match declaration argument name arg3"}
      INTEGER, INTENT(INOUT) :: ARG4
    END FUNCTION
END SUBMODULE

SUBMODULE (m:n) k
  CONTAINS
    MODULE SUBROUTINE sub2(arg3) !{error "PGF90-S-1057-Definition argument name arg3 does not match declaration argument name arg2"}
      INTEGER, intent(inout) :: arg3
      print *, arg3
    END SUBROUTINE sub2

    MODULE SUBROUTINE sub4(arg5) !{error "PGF90-S-1057-Definition argument name arg5 does not match declaration argument name arg4"}
      INTEGER, intent(inout) :: arg5
      print *, arg5
    END SUBROUTINE sub4

    INTEGER MODULE FUNCTION F2(ARG5) !{error "PGF90-S-1057-Definition argument name arg5 does not match declaration argument name arg4"}
      INTEGER, INTENT(INOUT) :: ARG5
    END FUNCTION

END SUBMODULE

