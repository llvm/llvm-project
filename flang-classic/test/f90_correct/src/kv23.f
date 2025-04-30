** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 
* Vectorizer - invariant expandable array reference bug discovered in
* MOLPRO.  An array reference is replaced by a scalar in the loop and
* the scalar is stored back into the array after the loop; however, there
* exists a conflicting reference to the same array in the loop.
*

	program test
	implicit real (a-h,o-z)
	dimension eta(3,3,2)
	common /parma/ mconu, mrcru
	parameter (NT=4)
	real expect(NT)
	real result(NT)

	mrcru = 3
	mconu = 3
	call fill(eta, 2.0, 18)

	call arinp(eta, 2, 3, 3, 8.0)

	result(1) = eta(1,1,1)
	result(2) = eta(1,2,1)
	result(3) = eta(1,1,2)
	result(4) = eta(1,2,2)
!	print 99, result
!99	format(9f8.1)
	data expect/1,1,1,1/
	call check(result, expect, NT)
	end
	subroutine arinp(eta,ncons,nrcru,nconu,a1)
	implicit real (a-h,o-z)
	common /parma/ mconu, mrcru
	dimension eta(mrcru,mconu,2)
	
cpgi$l novector
	do icons=1,ncons
!+++        +++ eta(1,1,icons) in the do212 loop is discovered to be invariant
!+++        +++    and replaced with the scalar .ndd000 and is initialized
!+++        +++    with:
!+++        +++ .ndd000 = eta(1,1,icons)
	    do 212 ircr=1,nrcru
		if(nconu.ne.1) go to 196
!+++                +++ replace the invariant array ref with .ndd000
!+++                +++ .ndd000 = a1
		    eta(1,1,icons)=a1
		go to 212
196	        continue
		do icon=1,nconu
!+++                +++  conflicting array reference
!+++                +++  !!!!!!!!!!!!!!!!!!!!!!!!!!!
		    eta(ircr,icon,icons)=eta(ircr,icon,icons)/2.0
		enddo
212	    continue
!+++        +++ add store back
!+++        +++ eta(1,1,icons) = .ndd000
	enddo
	end
	subroutine fill(aaa, a, n)
	real aaa(n)
cpgi$l novector
	do i = 1, n
	    aaa(i) = a
	enddo
	end
