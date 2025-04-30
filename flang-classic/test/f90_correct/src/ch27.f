!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! allocatable character array with deferred length

	program allocdeferredlen
	call sub
	end
	subroutine sub
        integer, parameter :: NT = 6
        integer, parameter :: N = 21
	integer*4 result(NT), expect(NT)
	data expect/48,-21,21,1,1,43/
        character(len=:), allocatable :: FileName(:)

        allocate ( character(len=48) :: FileName(-N:N) )
        FileName(:) = (/ ( char(64+k), k = -N, N ) /)
        result(1) = len(FileName)
        result(2) = lbound(FileName,1)
        result(3) = ubound(FileName,1)
	deallocate(FileName)

! ========   note the intersting allocatable assignment semantics here!!!
        allocate ( character(len=48) :: FileName(-N:N) )
        FileName = (/ ( char(64+k), k = -N, N ) /)
        result(4) = len(FileName)
        result(5) = lbound(FileName,1)
        result(6) = ubound(FileName,1)

!	write(0, '(6i4)') expect, result
	call check(result, expect, NT)

        end
