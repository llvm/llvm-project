!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! test namelist with specifier decimal comma

module numeric_
	implicit none

	integer, parameter :: INT_   = selected_int_kind(R=9)
	integer, parameter :: REAL_  = selected_real_kind(P=13,R=300)
	integer, parameter :: COMPLEX_  = selected_real_kind(P=13,R=300)

	interface show_kind
		module procedure show_int_, show_real_, show_complex_
	end interface

	private show_int_, show_real_, show_complex_

contains

subroutine show_int_(x)
	implicit none
	integer(kind=INT_), intent(in) :: x	! kind must not be variable

	write(6,'(/1x,"***INT_      KIND = ", i6, 10x,"requested = ", i6)') &
		kind(x), selected_int_kind(R=9)
	write(6,'(1x,"RADIX             = ", i6)') radix(x)
	write(6,'(1x,"DIGITS            = ", i6)') digits(x)
	write(6,'(1x,"RANGE             = ", i6)') range(x)
	write(6,'(1x,"HUGE              = ")',advance='NO')
	write(6, *)	huge(x)
	write(6,'(1x,"value             = ")',advance='NO')
	write(6, *)	x
end subroutine show_int_

subroutine show_real_(x)
	implicit none
	real(kind=REAL_), intent(in) :: x	! kind must not be variable

	write(6,'(/1x,"***REAL_     KIND = ", i6, 10x,"requested = ", i6)') &
		kind(x), selected_real_kind(P=13,R=300)
	write(6,'(1x,"PRECISION         = ", i6)') precision(x)
	write(6,'(1x,"MAXEXPONENT       = ", i6)') maxexponent(x)
	write(6,'(1x,"MINEXPONENT       = ", i6)') minexponent(x)
	write(6,'(1x,"RADIX             = ", i6)') radix(x)
	write(6,'(1x,"DIGITS            = ", i6)') digits(x)
	write(6,'(1x,"EPSILON           = ")',advance='NO')
	write(6, *)	epsilon(x)
	write(6,'(1x,"value             = ")',advance='NO')
	write(6, *)	x
end subroutine show_real_

subroutine show_complex_(x)
	implicit none
	complex(kind=COMPLEX_), intent(in) :: x	! kind must not be variable

	write(6,'(/1x,"***COMPLEX_  KIND = ", i6, 10x,"requested = ", i6)') &
		kind(x), selected_real_kind(P=13,R=300)
	write(6,'(1x,"PRECISION         = ", i6)') precision(real(x))
	write(6,'(1x,"MAXEXPONENT       = ", i6)') maxexponent(real(x))
	write(6,'(1x,"MINEXPONENT       = ", i6)') minexponent(real(x))
	write(6,'(1x,"RADIX             = ", i6)') radix(real(x))
	write(6,'(1x,"DIGITS            = ", i6)') digits(real(x))
	write(6,'(1x,"EPSILON           = ")',advance='NO')
	write(6, *)	epsilon(real(x))
	write(6,'(1x,"value             = ")',advance='NO')
	write(6, *)	x
end subroutine show_complex_

end module numeric_

module datastuff
	use numeric_
	implicit none

! for comparison purposes
	real ::		ttreal=1.0, 	treal = 1.0
	integer ::	ttinteger=2,	tinteger = 2
	complex ::	ttcomplex=(3.0,4.0),	tcomplex = (3.0, 4.0)
	character*10 ::	ttchar='namelist',	tchar = 'namelist'
	logical ::	ttbool=.TRUE.,		tbool = .TRUE.

	real, dimension(4) :: areal = (/ 1.0, 1.0, 2.0, 3.0 /)
	real, dimension(4) :: aareal = (/ 1.0, 1.0, 2.0, 3.0 /)
	integer, dimension(4) :: ainteger = (/ 2, 2, 3, 4 /)
	integer, dimension(4) :: aainteger = (/ 2, 2, 3, 4 /)
	complex, dimension(4) :: acomplex = (/ (3.0, 4.0), &
		(3.0, 4.0), (5.0, 6.0), (7.0, 7.0) /)
	complex, dimension(4) :: aacomplex = (/ (3.0, 4.0), &
		(3.0, 4.0), (5.0, 6.0), (7.0, 7.0) /)
	character*10, dimension(4) :: achar = (/ 'namelist  ', 'namelist  ',&
		'array     ', ' the lot  ' /)
	character*10, dimension(4) :: aachar = (/ 'namelist  ', 'namelist  ',&
		'array     ', ' the lot  ' /)
	logical, dimension(4) :: abool = (/ .TRUE., .TRUE., .FALSE., &
		.FALSE. /)
	logical, dimension(4) :: aabool = (/ .TRUE., .TRUE., .FALSE., &
		.FALSE. /)

	real(kind=REAL_) ::	xxreal= 1.0_REAL_,   xreal = 1.0_REAL_
	integer(kind=INT_) ::	xxinteger = 2_INT_,  xinteger = 2_INT_
	complex(kind=COMPLEX_) :: xcomplex = (3.0_COMPLEX_, 4.0_COMPLEX_)
	complex(kind=COMPLEX_) :: xxcomplex = (3.0_COMPLEX_, 4.0_COMPLEX_)
contains
	subroutine clearstuff()
		treal =	0.0
		tinteger =	0
		tcomplex =	(0.0,0.0)
		tchar =	''
		tbool =	.FALSE.
		areal(1:4) =	0.0
		ainteger(1:4) =0
		acomplex(1:4) =(0.0,0.0)
		achar(1:4) =	''
		abool(1:4) =	.FALSE.
		xreal =	0.0_REAL_
		xinteger =	0_INT_
		xcomplex =	(0.0_COMPLEX_,0.0_COMPLEX_)
	end subroutine clearstuff

	subroutine diffstuff(result, fullout)
	implicit none
	logical :: fullout
	integer :: numbad,i
	integer result(*)
! Compare the input data to the expected value

	numbad = 0
	if (ttreal .ne. treal) then
		numbad = numbad + 1
                result(1) = 1
		if (fullout) write(6,*) 'treal diff = ', ttreal - treal
	endif
	if (ttinteger .ne. tinteger) then
		numbad = numbad + 1
                result(2) = 1
		if (fullout) write(6,*) 'tinteger diff = ', ttinteger - tinteger
	endif
	if (ttcomplex .ne. tcomplex) then
		numbad = numbad + 1
                result(3) = 1
		if (fullout) write(6,*) 'tcomplex diff = ', ttcomplex - tcomplex
	endif
	if (ttchar .ne. tchar) then
		numbad = numbad + 1
                result(4) = 1
		if (fullout) write(6,*) 'tchar diff = ', ttchar, tchar
	endif
	if (ttbool .xor. tbool) then
		numbad = numbad + 1
                result(5) = 1
		if (fullout) write(6,*) 'tbool diff = ', ttbool, tbool
	endif

	do i = 1,4
		if (aareal(i) .ne. areal(i)) then
			numbad = numbad + 1
                        result(6) = 1
	if (fullout) write(6,*) 'areal diff = ', aareal(i) - areal(i)
		endif
		if (aainteger(i) .ne. ainteger(i)) then
                        result(7) = 1
			numbad = numbad + 1
	if (fullout) write(6,*) 'ainteger diff = ', aainteger(i) - ainteger(i)
		endif
		if (aacomplex(i) .ne. acomplex(i)) then
                        result(8) = 1
			numbad = numbad + 1
	if (fullout) write(6,*) 'acomplex diff = ', aacomplex(i) - acomplex(i)
		endif
		if (aachar(i) .ne. achar(i)) then
                        result(9) = 1
			numbad = numbad + 1
	if (fullout) write(6,*) 'achar diff = ', aachar(i), achar(i)
		endif
		if (aabool(i) .xor. abool(i)) then
                        result(10) = 1
			numbad = numbad + 1
	if (fullout) write(6,*) 'abool diff = ', aabool(i), abool(i)
		endif
	enddo

	if (xxreal .ne. xreal) then
		numbad = numbad + 1
                result(11) = 1
		if (fullout) write(6,*) 'xreal diff = ', xxreal - xreal
	endif
	if (xxinteger .ne. xinteger) then
		numbad = numbad + 1
                result(12) = 1
		if (fullout) write(6,*) 'xinteger diff = ', xxinteger - xinteger
	endif
	if (xxcomplex .ne. xcomplex) then
		numbad = numbad + 1
                result(13) = 1
		if (fullout) write(6,*) 'xcomplex diff = ', xxcomplex - xcomplex
	endif


!	if (numbad .ne. 0) then
!		write(6,'(1x,"found ",i3," differences")') numbad
!	else
!		write(6,'(1x,"found "," no"," differences")')
!	endif

	end subroutine diffstuff

end module datastuff

program tnmlist
	use datastuff
!	implicit none
        parameter(N=26)
        integer result(N), expect(N)
!
! demonstrate the f90 standard namelist
!
! Note that NAMELIST syntax is similar to COMMON BLOCKs
	namelist /tdata/ treal, tinteger, tcomplex, tchar, tbool
	namelist /adata/ areal, ainteger, acomplex, achar, abool
	namelist /xdata/ xreal, xinteger, xcomplex

	namelist /ttdata/ ttreal, ttinteger, ttcomplex, ttchar, ttbool
	namelist /aadata/ aareal, aainteger, aacomplex, aachar, aabool
	namelist /xxdata/ xxreal, xxinteger, xxcomplex

        expect = 0
        result = 0
        
! the OPEN statement defines many of the
! NAMELIST characteristics
	! need the delim, else some implementations will not surround
	! character strings with delimiters
	! recl limits the I/O to 80 character lines
!	open(6, recl=80, delim='APOSTROPHE')
	open(8,file="tnmlist.in", action='write', recl=80, delim='APOSTROPHE')
	!
! NAMELIST output varies with compilers
	! how NAMELIST data displays on your system
	write(8,nml=tdata)
	write(8,nml=adata)
	write(8,nml=xdata)
        close(8)

	open(9,file="tnmlist2.in", action='write', recl=80, delim='APOSTROPHE')
	write(9,nml=tdata, decimal='comma')
	write(9,nml=adata, decimal='comma')
	write(9,nml=xdata, decimal='comma')
        close(9)

	call clearstuff()
	open(8,file="tnmlist.in", status='OLD', recl=80, delim='APOSTROPHE')
!	write(6,*)
!	write(6,*) 'Read first batch'
	read(8,nml=tdata)
	read(8,nml=adata)
	read(8,nml=xdata)
        close(8)

	call diffstuff(result, fullout=.FALSE.)

!	write(6,*) 'Read second batch'
	call clearstuff()

	open(9,file="tnmlist2.in", status='OLD', recl=80, delim='APOSTROPHE')
	read(9,nml=tdata, decimal='comma')
	read(9,nml=adata, decimal='comma')
	read(9,nml=xdata, decimal='comma')
	call diffstuff(result(24), fullout=.FALSE.)

        call check(result, expect, N)

end program tnmlist

