!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This example is slightly modified from the Preprocessor chapter in the
! C99 spec: Example 6 in section 6.10.3.4:
! Test legal redefinition rules 
!
! ** Currently we pass comments (either macro-expanded or not) to the lexer...
! ** this will generate two warnings regarding macro-redefinition.
!
#define OBJ_LIKE (1-1)
#define OBJ_LIKE (1-1) !Comment
#define FUNC_LIKE(x) (x)
#define FUNC_LIKE(x) (x) !Comment
program p
    ! This test will only produce a warnings
    call check(.true., .true., 1)
end program
