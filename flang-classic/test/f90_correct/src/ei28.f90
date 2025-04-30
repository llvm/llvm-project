! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Initialization for logical and relational operators.

program ei28

parameter(NTEST=64)
integer :: result(NTEST)=1
integer :: expect(NTEST)=(/ &
1,1,0,1,0,0, &
1,1,0,1,0,0, &

!single relational real
1,1,0,1,0,0, &
1,1,0,1,0,0, &

!single relational character
0,0,0,1,1,1, &
0,0,0,1,1,1, &

!single relational double
1,1,0,1,0,0, &
1,1,0,1,0,0, &

!single logical
0,0,1,0, &
0,1,1,0, &

!single relational double complex
0,1, &
0,1, &

!single relational float complex
0,1, &
0,1 /)


!Following will be done in back end dinit
logical :: int7 = (ichar("a") == 97)
logical :: int8 = (ichar("a") >= 97)
logical :: int9 = (ichar("a") > 97)
logical :: int10 = (ichar("a") <= 97)
logical :: int11 = (ichar("a") < 97)
logical :: int12 = (ichar("a") .ne. 97)

logical,parameter :: int1 = (ichar("a") == 97)
logical,parameter :: int2 = (ichar("a") >= 97)
logical,parameter :: int3 = (ichar("a") > 97)
logical,parameter :: int4 = (ichar("a") <= 97)
logical,parameter :: int5 = (ichar("a") < 97)
logical,parameter :: int6 = (ichar("a") .ne. 97)

logical :: real7 = ((2.7**2) == 2.7**2)
logical :: real8 = ((2.7**2) >= 2.7**2)
logical :: real9 = ((2.7**2) > 2.7**2)
logical :: real10 = ((2.7**2) <= 2.7**2) 
logical :: real11 = ((2.7**2) < 2.7**2)
logical :: real12 = ((2.7**2) .ne. 2.7**2)

logical,parameter :: real1 = ((2.7**2) == 2.7**2)
logical,parameter :: real2 = ((2.7**2) >= 2.7**2)
logical,parameter :: real3 = ((2.7**2) > 2.7**2)
logical,parameter :: real4 = ((2.7**2) <= 2.7**2)
logical,parameter :: real5 = ((2.7**2) < 2.7**2)
logical,parameter :: real6 = ((2.7**2) .ne. 2.7**2)

logical :: char7 = 'a' == 'b'
logical :: char8 = 'a' >= 'b'
logical :: char9 = 'a' >  'b'
logical :: char10 = 'a' <= 'b'
logical :: char11 = 'a' <  'b'
logical :: char12 = 'a' .ne. 'b'

logical,parameter :: char1 = 'a' == 'b'
logical,parameter :: char2 = 'a' >= 'b'
logical,parameter :: char3 = 'a' >  'b'
logical,parameter :: char4 = 'a' <= 'b'
logical,parameter :: char5 = 'a' <  'b'
logical,parameter :: char6 = 'a' .ne. 'b'

logical,parameter :: double1 = ((2.7d0**2) == 2.7d0**2)
logical,parameter :: double2 = ((2.7d0**2) >= 2.7d0**2)
logical,parameter :: double3 = ((2.7d0**2) > 2.7d0**2)
logical,parameter :: double4 = ((2.7d0**2) <= 2.7d0**2)
logical,parameter :: double5 = ((2.7d0**2) < 2.7d0**2)
logical,parameter :: double6 = ((2.7d0**2) .ne. 2.7d0**2)

logical :: double7 = ((2.7d0**2) == 2.7d0**2)
logical :: double8 = ((2.7d0**2) >=2.7d0**2)
logical :: double9 = ((2.7d0**2) > 2.7d0**2)
logical :: double10 = ((2.7d0**2) <= 2.7d0**2)
logical :: double11 = ((2.7d0**2) < 2.7d0**2)
logical :: double12 = ((2.7d0**2) .ne. 2.7d0**2)

!front end
logical,parameter::land=(ichar("a")>=ichar("b").and.(.not.(ichar("c")<ichar("d"))))
logical,parameter::lor=(ichar("a")>=ichar("b").or.(.not.(ichar("c")<ichar("d"))))
logical,parameter::lneqv=(ichar("a")>=ichar("b").neqv.ichar("c")<ichar("d"))
logical,parameter::leqv=(ichar("a")>=ichar("b").eqv.ichar("c")<ichar("d"))

!back end
logical::land2=(ichar("a")>=ichar("b").and.(.not.(ichar("c")<ichar("d"))))
logical::lor2=(ichar("a")>=ichar("b").or.ichar("c")<ichar("d"))
logical::lneqv2=(ichar("a")>=ichar("b").neqv.ichar("c")<ichar("d"))
logical::leqv2=(ichar("a")>=ichar("b").eqv.ichar("c")<ichar("d"))

logical,parameter :: cmplx1 = (1.0d0,-1.0d0) == (-1.0d0,-1.0d0)
logical,parameter :: cmplx2 = (1.0d0,-1.0d0) .ne. (-1.0d0,-1.0d0)

logical :: cmplx3 = (1.0d0,-1.0d0) == (-1.0d0,-1.0d0)
logical :: cmplx4 = (1.0d0,-1.0d0) .ne. (-1.0d0,-1.0d0)

logical :: cmplxf1 = ((1.0,-1.0) == (-1.0,-1.0))
logical :: cmplxf2 = ((1.0,-1.0) .ne. (-1.0,-1.0))

logical,parameter :: cmplxf3 = ((1.0,-1.0) == (-1.0,-1.0))
logical,parameter :: cmplxf4 = ((1.0,-1.0) .ne. (-1.0,-1.0))

if (int1 == .FALSE.) then
    result(1) = 0
endif
if (int2 == .FALSE.) then
    result(2) = 0
endif
if (int3 == .FALSE.) then
    result(3) = 0
endif
if (int4 == .FALSE.) then
    result(4) = 0
endif
if (int5 == .FALSE.) then
    result(5) = 0
endif
if (int6 == .FALSE.) then
    result(6) = 0
endif
if (int7 == .FALSE.) then
    result(7) = 0
endif
if (int8 == .FALSE.) then
    result(8) = 0
endif
if (int9 == .FALSE.) then
    result(9) = 0
endif
if (int10 == .FALSE.) then
    result(10) = 0
endif
if (int11 == .FALSE.) then
    result(11) = 0
endif
if (int12 == .FALSE.) then
    result(12) = 0
endif

if (real1 == .FALSE.) then
    result(13) = 0
endif
if (real2 == .FALSE.) then
    result(14) = 0
endif
if (real3 == .FALSE.) then
    result(15) = 0
endif
if (real4 == .FALSE.) then
    result(16) = 0
endif
if (real5 == .FALSE.) then
    result(17) = 0
endif
if (real6 == .FALSE.) then
    result(18) = 0
endif
if (real7 == .FALSE.) then
    result(19) = 0
endif
if (real8 == .FALSE.) then
    result(20) = 0
endif
if (real9 == .FALSE.) then
    result(21) = 0
endif
if (real10 == .FALSE.) then
    result(22) = 0
endif
if (real11 == .FALSE.) then
    result(23) = 0
endif
if (real12 == .FALSE.) then
    result(24) = 0
endif

if (char1 == .FALSE.) then
    result(25) = 0
endif
if (char2 == .FALSE.) then
    result(26) = 0
endif
if (char3 == .FALSE.) then
    result(27) = 0
endif
if (char4 == .FALSE.) then
    result(28) = 0
endif
if (char5 == .FALSE.) then
    result(29) = 0
endif
if (char6 == .FALSE.) then
    result(30) = 0
endif
if (char7 == .FALSE.) then
    result(31) = 0
endif
if (char8 == .FALSE.) then
    result(32) = 0
endif
if (char9 == .FALSE.) then
    result(33) = 0
endif
if (char10 == .FALSE.) then
    result(34) = 0
endif
if (char11 == .FALSE.) then
    result(35) = 0
endif
if (char12 == .FALSE.) then
    result(36) = 0
endif

if (double1 == .FALSE.) then
    result(37) = 0
endif
if (double2 == .FALSE.) then
    result(38) = 0
endif
if (double3 == .FALSE.) then
    result(39) = 0
endif
if (double4 == .FALSE.) then
    result(40) = 0
endif
if (double5 == .FALSE.) then
    result(41) = 0
endif
if (double6 == .FALSE.) then
    result(42) = 0
endif
if (double7 == .FALSE.) then
    result(43) = 0
endif
if (double8 == .FALSE.) then
    result(44) = 0
endif
if (double9 == .FALSE.) then
    result(45) = 0
endif
if (double10 == .FALSE.) then
    result(46) = 0
endif
if (double11 == .FALSE.) then
    result(47) = 0
endif
if (double12 == .FALSE.) then
    result(48) = 0
endif

if (land == .FALSE.) then
    result(49) = 0
endif
if (lor == .FALSE.) then
    result(50) = 0
endif
if (lneqv == .FALSE.) then
    result(51) = 0
endif
if (leqv == .FALSE.) then
    result(52) = 0
endif
if (land2 == .FALSE.) then
    result(53) = 0
endif
if (lor2 == .FALSE.) then
    result(54) = 0
endif
if (lneqv2 == .FALSE.) then
    result(55) = 0
endif
if (leqv2 == .FALSE.) then
    result(56) = 0
endif

if (cmplx1 == .FALSE.) then
    result(57) = 0
endif
if (cmplx2 == .FALSE.) then
    result(58) = 0
endif
if (cmplx3 == .FALSE.) then
    result(59) = 0
endif
if (cmplx4 == .FALSE.) then
    result(60) = 0
endif

if (cmplxf1 == .FALSE.) then
    result(61) = 0
endif
if (cmplxf2 == .FALSE.) then
    result(62) = 0
endif
if (cmplxf3 == .FALSE.) then
    result(63) = 0
endif
if (cmplxf4 == .FALSE.) then
    result(64) = 0
endif

call check(result,expect, NTEST)

end

