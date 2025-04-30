!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
C	This tests directives
!	And also comments
*	Even if comments or directives begin with an astrisk
c	And also macro replacement within directives
	program p
	logical :: res(1) = .false., expect(1) = .true.

#define FOO      critical
#define FN(c, d) c d
#define FIN      end
#define STR(_x) #_x
#define CPY(_x) _x

!$omp FOO
	print *, CPY("Making a critical block! WooHoo!")
	print *, STR(Make statements parallel! WooHoo!)
C$omp end FOO

*$omp FOO
	print *, "Many moar things!"
c$omp FN(FIN,FOO)
	res(1) = .true.
	call check(res, expect, 1)
	end
