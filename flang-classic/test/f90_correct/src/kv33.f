C Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
C See https://llvm.org/LICENSE.txt for license information.
C SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

	double precision a0(100), a1(100)
	double precision b0(100), b1(100)

	do i = 1,100
	 a0(i) = i
	 a1(i) = 3
	 b0(i) = 0
	 b1(i) = 0
	 b1(i) = mod(a0(i),a1(i))
	 b0(i) = mod(a0(i),a1(i))
	enddo

	call checkd( b0, b1, 100 )
	end
