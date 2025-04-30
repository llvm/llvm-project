!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!   KIND with various basic datatypes

program p
 real(4) a
 real(8) b
 complex(4) c
 complex(8) d
 integer(1) e
 integer(2) f
 integer(4) g
 integer(8) h
 integer result(8)
 integer expect(8)
 data expect/4,8,4,8,1,2,4,8/
 result(1) = kind(a)
 result(2) = kind(b)
 result(3) = kind(c)
 result(4) = kind(d)
 result(5) = kind(e)
 result(6) = kind(f)
 result(7) = kind(g)
 result(8) = kind(h)
! print *,result
 call check(result,expect,8)
end program
