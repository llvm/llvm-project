** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   adjustable POINTER object and ALLOCATE statement

	parameter (N=1)
	integer result(N)
	integer expect(N)

  	maxdat=1000
	call glim3(maxdat, result)

	data expect /1/
	call check(result, expect, N)

	end
	subroutine glim3(imxdat, ires)
	double precision sspmat
	integer imxdat
	pointer (l_sspmat, sspmat(imxdat))

c  ensure ilms are written for sspmat's array descriptor 

	data kount/0/, l2dir/0/	 ! cause ilms to be erased

	maxdat=imxdat
	allocate(sspmat, stat=jallo)
	if ( jallo .eq. 0 ) goto 100
	ires = 2
	goto 9999
 100    do 18 i=1,maxdat
	   sspmat(i)=0.d0
  18    continue
	ires = 1
	deallocate(sspmat)
 9999   continue
	return
	end
