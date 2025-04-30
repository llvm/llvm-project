!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests define and undef macros 
!
#define FOO
#undef FOO
#define BAR 42
#define BAZ 43

program p
! Test undef
#ifdef FOO
    This will cause a compiler error
#else
    print *, "Works!"
#endif

! Test defined operator using BAR and BAZ 
#if defined(BAR) && defined(BAZ) && (BAR < BAZ)
    print *, "Works!"
#elif defined(FOO)
    This will cause a compiler error
#else
    ... So will this
#endif

! Test != vs comment
# if BAR != BAZ
    print *, "Works!"
# else
    This will cause a compiler error
# endif

    call check(.true., .true., 1)
end program
