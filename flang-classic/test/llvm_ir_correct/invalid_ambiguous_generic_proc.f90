! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! RUN: %flang -S -emit-llvm %s

MODULE dbcsr_ambiguous_operations
  IMPLICIT NONE
  PUBLIC :: dbcsr_ambiguous_add
  PRIVATE
  INTERFACE dbcsr_ambiguous_add
    MODULE PROCEDURE dbcsr_ambiguous_add_d
  END INTERFACE
CONTAINS
  SUBROUTINE dbcsr_ambiguous_add_d()
    print *, "add_d and anytype."
  END SUBROUTINE dbcsr_ambiguous_add_d
END MODULE dbcsr_ambiguous_operations

MODULE dbcsr_ambiguous_api
  USE dbcsr_ambiguous_operations,ONLY: &
    dbcsr_ambiguous_add_prv=>dbcsr_ambiguous_add
  IMPLICIT NONE
  PUBLIC :: dbcsr_ambiguous_add
  PRIVATE
  INTERFACE dbcsr_ambiguous_add
    MODULE PROCEDURE dbcsr_ambiguous_add_dd
  END INTERFACE
  CONTAINS
    SUBROUTINE dbcsr_ambiguous_add_dd()
      print *, "add_dd in API."
    END SUBROUTINE
END MODULE dbcsr_ambiguous_api

PROGRAM dbcsr_ambiguous_test_csr_conversions
USE dbcsr_ambiguous_api,ONLY:  dbcsr_ambiguous_add
IMPLICIT NONE
  CONTAINS
  SUBROUTINE csr_conversion_test()
    CALL dbcsr_ambiguous_add()
  END SUBROUTINE csr_conversion_test
END PROGRAM dbcsr_ambiguous_test_csr_conversions

