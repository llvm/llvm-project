!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! RUN: %flang %s

PROGRAM charlen
END PROGRAM charlen
FUNCTION cksum(data)
CHARACTER(LEN=30):: cksum
CHARACTER(KIND=2):: data(:)
END
