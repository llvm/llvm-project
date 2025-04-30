!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

	subroutine check(result, expect, np)
	integer np
	integer result(np)
	integer expect(np)
	integer tests_passed
	integer tests_failed
	tests_passed = 0
	tests_failed = 0
	do i = 1,np
	    if (expect(i) .ne. result(i)) then
		tests_failed = tests_failed + 1
		print 1,i,result(i),result(i),expect(i),expect(i)
1	format(' --- test number ',i5,' FAILED. res',i5,'(',z8.8,')',
     &		'  exp',i5,'(',z8.8,')')
	    else
		tests_passed = tests_passed + 1
	    endif
	enddo
	if (tests_failed .eq. 0) then
	    print 2,np,tests_passed,tests_failed
	else
	    print 3,np,tests_passed,tests_failed
	endif
2	format(' --- ',i3,' tests completed.',i5,' tests PASSED.',
     &		i5,' tests failed.')
3	format(' --- ',i3,' tests completed.',i5,' tests passed.',
     &		i5,' tests FAILED.')
	end
	subroutine checkd(result, expect, np)
	integer np
	real*8 result(np)
	real*8 expect(np)
	integer tests_passed
	integer tests_failed
	tests_passed = 0
	tests_failed = 0
	do i = 1,np
	    if (expect(i) .ne. result(i)) then
		tests_failed = tests_failed + 1
		print 1,i,result(i),expect(i)
1	format(' --- test number ',i5,' FAILED. res',g12.8,
     &			'  exp',g12.8)
	    else
		tests_passed = tests_passed + 1
	    endif
	enddo
	if (tests_failed .eq. 0) then
	    print 2,np,tests_passed,tests_failed
	else
	    print 3,np,tests_passed,tests_failed
	endif
2	format(' --- ',i3,' tests completed.',i5,' tests PASSED.',
     &		i5,' tests failed.')
3	format(' --- ',i3,' tests completed.',i5,' tests passed.',
     &		i5,' tests FAILED.')
	end
