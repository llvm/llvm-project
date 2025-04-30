!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Conditional compilation sentinels
	program test
	i = 0
!$      i = i + 1
*$      i = i + 10
c$      i = i + 100
C$      i = i + 1000
	call check(i,1111,1)
	end
