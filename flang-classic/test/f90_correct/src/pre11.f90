!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests replacement and concatenation with single and 
! double quote literals.
!
#define STR(_s) _s
#define CAT(_a, _b) _a##_b

program p
    logical :: res(2), expect(2) = (/.true., .true./)
    print *, STR('c character ! bang " quote * asterisk') ! Single-quote
    res(1) = STR('c character ! bang " quote * asterisk') .eq. 'c character ! bang " quote * asterisk'
        
    print *, STR("c character ! bang ' quote * asterisk") ! Double-quote
    res(2) = STR("c character ! bang ' quote * asterisk") .eq. "c character ! bang ' quote * asterisk"

    print *, CAT('! bang " quote * asterisk', '! bang " quote * asterisk') ! Single-quote
    print *, CAT("! bang ' quote * asterisk", "! bang ' quote * asterisk") ! Double-quote

    call check(res, expect,  2)
end program
