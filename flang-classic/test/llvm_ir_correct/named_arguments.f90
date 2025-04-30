! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! RUN: %flang -S -emit-llvm %s

MODULE operations_module
   IMPLICIT NONE
   PUBLIC :: operation
   PRIVATE
   INTERFACE operation
      MODULE PROCEDURE operation_anytype
   END INTERFACE
CONTAINS
   SUBROUTINE operation_anytype(key)
      REAL(8), INTENT(IN)                :: key
   END SUBROUTINE operation_anytype
END MODULE operations_module

MODULE api_module
   USE operations_module, ONLY: operation_prv => operation
   IMPLICIT NONE
   PUBLIC :: operation
   PRIVATE
CONTAINS
   SUBROUTINE operation(key)
      REAL(8), INTENT(IN)                               :: key
   END SUBROUTINE operation
END MODULE api_module

MODULE methods_module
   USE api_module, ONLY: operation
   IMPLICIT NONE
CONTAINS
   SUBROUTINE method(key_value)
      REAL(KIND = 8), INTENT(IN)                          :: key_value
      CALL operation(key = key_value)
   END SUBROUTINE method
END MODULE methods_module
