! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! RUN: %flang -S -emit-llvm %s

MODULE dbcsr_public_operations
 PUBLIC :: dbcsr_public_norm
 INTERFACE dbcsr_public_norm
  MODULE PROCEDURE dbcsr_public_norm_anytype
 END INTERFACE
 CONTAINS
 SUBROUTINE dbcsr_public_norm_anytype
   print *, "hello world"
 END SUBROUTINE
END MODULE dbcsr_public_operations

MODULE dbcsr_public_api
use dbcsr_public_operations
 use dbcsr_public_operations, only: &
 dbcsr_public_norm_prv => dbcsr_public_norm
 public::dbcsr_public_norm
private
 CONTAINS
 SUBROUTINE dbcsr_public_norm()
  print *, "hellow world from dbcsr_public_api "
 end subroutine
end module

program main
use dbcsr_public_api, only : dbcsr_public_norm
 call dbcsr_public_norm();
end program
