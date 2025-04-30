!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

c	Simple OpenMP Parallel Region

	program p
	call t1
	end

	subroutine t1
	 integer atomic, res
	 integer act(2), expt(2)
	 atomic = 0
	 res = 0
c$omp	parallel shared(atomic, res)
c$omp	atomic update
	 atomic = atomic + 1

c$omp	end parallel

	 expt(1) = 2
	 act(1) = atomic
	 call check(act, expt, 1)

	end
