** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Register allocation bug when passing arguments by value to a
*   STDCALL *   routine on win32.  Even though STDCALL is only available on
*   win32, this test can still be used on other systems; there will not be
*   any value passing, but that's ok.

        integer function ifoo( hInstance, nCmdShow)
!DEC$ ATTRIBUTES STDCALL :: ifoo
        integer hInstance
        integer nCmdShow
        ii = hinstance
        jj = ncmdshow
        call bar(ii,jj)
        ifoo = 0
        end
        subroutine bar(ii,jj)
	common/ires/ires(2)
	ires(1) = ii
	ires(2) = jj
        end
	common/ires/ires(2)
	integer iexp(2)
	data iexp/1,2/
        integer ifoo
        external ifoo
!DEC$ ATTRIBUTES STDCALL :: ifoo
        kk = ifoo(1,2)
	call check(ires, iexp, 2)
        end

