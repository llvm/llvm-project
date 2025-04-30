!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for the import module from the correct path. This file is in the
! correct path.

module import_source2
    use import_source, only : func, sub1
end
