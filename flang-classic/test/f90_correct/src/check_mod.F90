!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module check_mod
  use ieee_arithmetic

  interface check
    module procedure checki1, checki2, checki4, checki8
    module procedure checkl1, checkl2, checkl4, checkl8
    module procedure checkr4, checkr8
#ifdef __flang_quadfp__
    module procedure checkr16
#endif
    module procedure checkc4, checkc8, checkc1
#ifdef __flang_quadfp__
    module procedure checkc16
#endif
    module procedure checkcptr, checkcptr2d, checkbytes, checkdt
  end interface

  interface ulperror
    module procedure ulperrorr4r4, ulperrorr4r8, ulperrorr8r8
#ifdef __flang_quadfp__
    module procedure ulperrorr16r16
#endif
  end interface

  interface ieeecheckcases
    module procedure ieeecheckcasesr4, ieeecheckcasesr8
#ifdef __flang_quadfp__
    module procedure ieeecheckcasesr16
#endif
  end interface

  integer, parameter :: maxulperror = 1000

  character(80) :: fmt00="('Relative Tolerance is ignored for integral checks')"
  character(80) :: fmt01="('Absolute and Relative Tolerance ignored for logical/character checks')"
  character(80) :: fmt02="('test number ',i0,' tolerated ',"
  character(80) :: fmt03="('test number ',i0,' FAILED. ',"
  character(80) :: fmt04="(i0,' tests completed. ',i0,' tests PASSED. 0 tests failed.')"
  character(80) :: fmt05="(i0,' tests completed. ',i0,' tests PASSED. ',i0,' tests tolerated \n PASS ')"
  character(80) :: fmt06="(i0,' tests completed. ',i0,' tests passed. ',i0,' tests FAILED.')"
  character(80) :: fmt07="('type ',a,', test number ',i0,', byte ',i0,' FAILED. ',"
  character(80) :: fmt08="('type ',a,', test number ',i0,', member ',i0,', byte ',i0,' FAILED. ',"

  character(80) :: fmt10="'res ',l0,' exp ',l0)"
  character(80) :: fmt11="'res ',i0,' (0x',z2.2,') exp ',i0,' (0x',z2.2,')')"
  character(80) :: fmt12="'res ',i0,' (0x',z4.4,') exp ',i0,' (0x',z4.4,')')"
  character(80) :: fmt13="'res ',i0,' (0x',z8.8,') exp ',i0,' (0x',z8.8,')')"
  character(80) :: fmt14="'res ',i0,' (0x',z16.16,') exp',i0,' (0x',z16.16,')')"
  character(80) :: fmt16="'res ',f0.3,' (0x',z8.8,') exp ',f0.3,' (0x',z8.8,')')"
  character(80) :: fmt17="'res ',f0.3,' (0x',z16.16,') exp ',f0.3,' (0x',z16.16,')')"
  character(80) :: fmt18="'res ',2(f0.3,1x),2('(0x',z8.8,') '),'exp ',2(f0.3,1x),2('(0x',z8.8,') '))"
  character(80) :: fmt19="'res ',2(f0.3,1x),2('(0x',z16.16,') '),'exp ',2(f0.3,1x),2('(0x',z16.16,') '))"

  character(80) :: fmt20="'res (0x',z2.2,') exp (0x',z2.2,')')"
#ifdef __flang_quadfp__
  character(160) :: fmt21="'res ',f0.33,' (0x',z33.33,') exp ',f0.33,' (0x',z33.33,')')"
  character(320) :: fmt22="'res ',2(f0.33,1x),2('(0x',z33.33,') '),'exp ',2(f0.33,1x),2('(0x',z33.33,') '))"
#endif

  contains
      real*4 function ulperrorr4r4(rc, dc)
      real*4 rc, dc
      if (rc.eq.0.0) then
        if (dc.eq.0.0) then
          ulperrorr4r4 = 0.0
        else
          ulperrorr4r4 = maxulperror
        endif
      else
        iexp = 23 - (exponent(rc) - 1)
        ulperrorr4r4 = abs(ieee_scalb(rc,iexp) - ieee_scalb(dc,iexp))
      end if
      return
      end

      real*4 function ulperrorr4r8(rc, dc)
      real*4 rc
      real*8 dc
      if (rc.eq.0.0) then
        if (dabs(dc) .lt. dble(ieee_next_after(0.0,1.0))) then
          ulperrorr4r8 = 0.0
        else
          ulperrorr4r8 = maxulperror
        endif
      else
        iexp = 23 - (exponent(rc) - 1)
        ulperrorr4r8 = real(dabs(ieee_scalb(dble(rc),iexp)-ieee_scalb(dc,iexp)))
      end if
      return
      end

      real*4 function ulperrorr8r8(rc, dc)
      real*8 rc, dc
      if (rc.eq.0.0d0) then
        if (dc.eq.0.0d0) then
          ulperrorr8r8 = 0.0
        else
          ulperrorr8r8 = maxulperror
        endif
      else
        iexp = 52 - (exponent(rc) - 1)
        ulperrorr8r8 = real(dabs(ieee_scalb(rc,iexp)-ieee_scalb(dc,iexp)))
      end if
      return
      end

#ifdef __flang_quadfp__
      real*4 function ulperrorr16r16(rc, dc)
      real*16 rc, dc
      if (rc.eq.0.0_16) then
        if (dc.eq.0.0_16) then
          ulperrorr16r16 = 0.0
        else
          ulperrorr16r16 = maxulperror
        endif
      else
        iexp = 112 - (exponent(rc) - 1)
        ulperrorr16r16 = real(abs(ieee_scalb(rc,iexp)-ieee_scalb(dc,iexp)))
      end if
      return
      end
#endif

      integer function ieeecheckcasesr4(xres, xexp)
      real*4 xres, xexp
      type(ieee_class_type) :: creslt, cexpct
      creslt = ieee_class(xres)
      cexpct = ieee_class(xexp)
      if (ieee_is_finite(xres).and.ieee_is_finite(xexp)) then
          ! Only check for +/- zero, else fall through
          if ((creslt.eq.ieee_positive_zero) .and. &
              (cexpct.ne.ieee_positive_zero)) then
              ieeecheckcasesr4 = 1  ! FAIL
          else if ((creslt.eq.ieee_negative_zero) .and. &
                   (cexpct.ne.ieee_negative_zero)) then
              ieeecheckcasesr4 = 1  ! FAIL
          else if (xres .eq. xexp) then
              ieeecheckcasesr4 = 2  ! PASS
          else
              ieeecheckcasesr4 = 0  ! Fail, check tolerances
          end if
      else if (creslt .ne. cexpct) then
          ieeecheckcasesr4 = 1   ! FAIL
      else
          ieeecheckcasesr4 = 2   ! PASS
      end if
      return
      end

      integer function ieeecheckcasesr8(xres, xexp)
      real*8 xres, xexp
      type(ieee_class_type) :: creslt, cexpct
      creslt = ieee_class(xres)
      cexpct = ieee_class(xexp)
      if (ieee_is_finite(xres).and.ieee_is_finite(xexp)) then
          ! Only check for +/- zero, else fall through
          if ((creslt.eq.ieee_positive_zero) .and. &
              (cexpct.ne.ieee_positive_zero)) then
              ieeecheckcasesr8 = 1  ! FAIL
          else if ((creslt.eq.ieee_negative_zero) .and. &
                   (cexpct.ne.ieee_negative_zero)) then
              ieeecheckcasesr8 = 1  ! FAIL
          else if (xres .eq. xexp) then
              ieeecheckcasesr8 = 2  ! PASS
          else
              ieeecheckcasesr8 = 0  ! Fail, check tolerances
          end if
      else if (creslt .ne. cexpct) then
          ieeecheckcasesr8 = 1   ! FAIL
      else
          ieeecheckcasesr8 = 2   ! PASS
      end if
      return
      end

#ifdef __flang_quadfp__
      integer function ieeecheckcasesr16(xres, xexp)
      real*16 xres, xexp
      type(ieee_class_type) :: creslt, cexpct
      creslt = ieee_class(xres)
      cexpct = ieee_class(xexp)
      if (ieee_is_finite(xres).and.ieee_is_finite(xexp)) then
          ! Only check for +/- zero, else fall through
          if ((creslt.eq.ieee_positive_zero) .and. &
              (cexpct.ne.ieee_positive_zero)) then
              ieeecheckcasesr16 = 1  ! FAIL
          else if ((creslt.eq.ieee_negative_zero) .and. &
                   (cexpct.ne.ieee_negative_zero)) then
              ieeecheckcasesr16 = 1  ! FAIL
          else if (xres .eq. xexp) then
              ieeecheckcasesr16 = 2  ! PASS
          else
              ieeecheckcasesr16 = 0  ! Fail, check tolerances
          end if
      else if (creslt .ne. cexpct) then
          ieeecheckcasesr16 = 1   ! FAIL
      else
          ieeecheckcasesr16 = 2   ! PASS
      end if
      return
      end
#endif

    ! First integer*1
    subroutine checki1(reslt, expct, np, atoler, rtoler)
!dir$ ignore_tkr (r) reslt, expct
      integer*1, dimension(*) :: reslt
      integer*1, dimension(*) :: expct
      integer :: np
      integer*1, optional :: atoler, rtoler
      integer*1 :: atol, rtol
      integer i, tests_passed, tests_failed, tests_tolerated
      if (present(atoler)) then
        atol = abs(atoler)
      else
        atol = 0
      endif
      if (present(rtoler)) then
        rtol = abs(rtoler)
      else
        rtol = 0
      endif
      if (rtol .gt. 0) write(6,fmt=fmt00)
      tests_passed = 0
      tests_failed = 0
      tests_tolerated = 0
      do i = 1, np
        if (expct(i) .eq. reslt(i)) then
            tests_passed = tests_passed + 1
        else if ((atol .gt. 0) .and. (abs(expct(i) - reslt(i)) .le. atol)) then
            tests_passed = tests_passed + 1
            tests_tolerated = tests_tolerated + 1
            if (tests_tolerated .le. 100) then
              write(6,fmt=fmt02//fmt11) i, reslt(i), reslt(i), expct(i), expct(i)
            end if
        else
            tests_failed = tests_failed + 1
            if (tests_failed .le. 100) then
              write(6,fmt=fmt03//fmt11) i, reslt(i), reslt(i), expct(i), expct(i)
            end if
        endif
      enddo
      if (tests_failed .eq. 0) then
         if (tests_tolerated .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
         else
            write(6,fmt=fmt05) np, tests_passed, tests_tolerated
         endif
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      endif
      return
    end subroutine checki1

    ! integer*2
    subroutine checki2(reslt, expct, np, atoler, rtoler)
!dir$ ignore_tkr (r) reslt, expct
      integer*2, dimension(*) :: reslt
      integer*2, dimension(*) :: expct
      integer :: np
      integer*2, optional :: atoler, rtoler
      integer*2 :: atol, rtol
      integer i, tests_passed, tests_failed, tests_tolerated
      if (present(atoler)) then
        atol = abs(atoler)
      else
        atol = 0
      endif
      if (present(rtoler)) then
        rtol = abs(rtoler)
      else
        rtol = 0
      endif
      if (rtol .gt. 0) write(6,fmt=fmt00)
      tests_passed = 0
      tests_failed = 0
      tests_tolerated = 0
      do i = 1, np
        if (expct(i) .eq. reslt(i)) then
            tests_passed = tests_passed + 1
        else if ((atol .gt. 0) .and. (abs(expct(i) - reslt(i)) .le. atol)) then
            tests_passed = tests_passed + 1
            tests_tolerated = tests_tolerated + 1
            if (tests_tolerated .le. 100) then
              write(6,fmt=fmt02//fmt12) i, reslt(i), reslt(i), expct(i), expct(i)
            end if
        else
            tests_failed = tests_failed + 1
            if (tests_failed .le. 100) then
              write(6,fmt=fmt03//fmt12) i, reslt(i), reslt(i), expct(i), expct(i)
            end if
        endif
      enddo
      if (tests_failed .eq. 0) then
         if (tests_tolerated .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
         else
            write(6,fmt=fmt05) np, tests_passed, tests_tolerated
         endif
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      endif
      return
    end subroutine checki2

    ! integer*4
    subroutine checki4(reslt, expct, np, atoler, rtoler)
!dir$ ignore_tkr (r) reslt, expct
      integer*4, dimension(*) :: reslt
      integer*4, dimension(*) :: expct
      integer :: np
      integer*4, optional :: atoler, rtoler
      integer*4 :: atol, rtol
      integer i, tests_passed, tests_failed, tests_tolerated
      if (present(atoler)) then
        atol = abs(atoler)
      else
        atol = 0
      endif
      if (present(rtoler)) then
        rtol = abs(rtoler)
      else
        rtol = 0
      endif
      if (rtol .gt. 0) write(6,fmt=fmt00)
      tests_passed = 0
      tests_failed = 0
      tests_tolerated = 0
      do i = 1, np
        if (expct(i) .eq. reslt(i)) then
            tests_passed = tests_passed + 1
        else if ((atol .gt. 0) .and. (abs(expct(i) - reslt(i)) .le. atol)) then
            tests_passed = tests_passed + 1
            tests_tolerated = tests_tolerated + 1
            if (tests_tolerated .le. 100) then
              write(6,fmt=fmt02//fmt13) i, reslt(i), reslt(i), expct(i), expct(i)
            end if
        else
            tests_failed = tests_failed + 1
            if (tests_failed .le. 100) then
              write(6,fmt=fmt03//fmt13) i, reslt(i), reslt(i), expct(i), expct(i)
            end if
        endif
      enddo
      if (tests_failed .eq. 0) then
         if (tests_tolerated .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
         else
            write(6,fmt=fmt05) np, tests_passed, tests_tolerated
         endif
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      endif
      return
    end subroutine checki4

    ! integer*8
    subroutine checki8(reslt, expct, np, atoler, rtoler)
!dir$ ignore_tkr (r) reslt, expct
      integer*8, dimension(*) :: reslt
      integer*8, dimension(*) :: expct
      integer :: np
      integer*8, optional :: atoler, rtoler
      integer*8 :: atol, rtol
      integer i, tests_passed, tests_failed, tests_tolerated
      if (present(atoler)) then
        atol = abs(atoler)
      else
        atol = 0
      endif
      if (present(rtoler)) then
        rtol = abs(rtoler)
      else
        rtol = 0
      endif
      if (rtol .gt. 0) write(6,fmt=fmt00)
      tests_passed = 0
      tests_failed = 0
      tests_tolerated = 0
      do i = 1, np
        if (expct(i) .eq. reslt(i)) then
            tests_passed = tests_passed + 1
        else if ((atol .gt. 0) .and. (abs(expct(i) - reslt(i)) .le. atol)) then
            tests_passed = tests_passed + 1
            tests_tolerated = tests_tolerated + 1
            if (tests_tolerated .le. 100) then
              write(6,fmt=fmt02//fmt14) i, reslt(i), reslt(i), expct(i), expct(i)
            end if
        else
            tests_failed = tests_failed + 1
            if (tests_failed .le. 100) then
              write(6,fmt=fmt03//fmt14) i, reslt(i), reslt(i), expct(i), expct(i)
            end if
        endif
      enddo
      if (tests_failed .eq. 0) then
         if (tests_tolerated .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
         else
            write(6,fmt=fmt05) np, tests_passed, tests_tolerated
         endif
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      endif
      return
    end subroutine checki8

    ! Now logical*1
    subroutine checkl1(reslt, expct, np, atoler, rtoler)
!dir$ ignore_tkr (r) reslt, expct
      logical*1, dimension(*) :: reslt
      logical*1, dimension(*) :: expct
      integer :: np
      logical*1, optional :: atoler, rtoler
      integer i, tests_passed, tests_failed
      if (present(atoler) .or. present(rtoler)) then
        write(6,fmt=fmt01)
      endif
      tests_passed = 0
      tests_failed = 0
      do i = 1, np
        if (expct(i) .eqv. reslt(i)) then
            tests_passed = tests_passed + 1
        else
            tests_failed = tests_failed + 1
            if (tests_failed .le. 100) then
              write(6,fmt=fmt03//fmt10) i, reslt(i), expct(i)
            end if
        endif
      enddo
      if (tests_failed .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      endif
      return
    end subroutine checkl1

    ! logical*2
    subroutine checkl2(reslt, expct, np, atoler, rtoler)
!dir$ ignore_tkr (r) reslt, expct
      logical*2, dimension(*) :: reslt
      logical*2, dimension(*) :: expct
      integer :: np
      logical*2, optional :: atoler, rtoler
      integer i, tests_passed, tests_failed
      if (present(atoler) .or. present(rtoler)) then
        write(6,fmt=fmt01)
      endif
      tests_passed = 0
      tests_failed = 0
      do i = 1, np
        if (expct(i) .eqv. reslt(i)) then
            tests_passed = tests_passed + 1
        else
            tests_failed = tests_failed + 1
            if (tests_failed .le. 100) then
              write(6,fmt=fmt03//fmt10) i, reslt(i), expct(i)
            end if
        endif
      enddo
      if (tests_failed .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      endif
      return
    end subroutine checkl2

    ! logical*4
    subroutine checkl4(reslt, expct, np, atoler, rtoler)
!dir$ ignore_tkr (r) reslt, expct
      logical*4, dimension(*) :: reslt
      logical*4, dimension(*) :: expct
      integer :: np
      logical*4, optional :: atoler, rtoler
      integer i, tests_passed, tests_failed
      if (present(atoler) .or. present(rtoler)) then
        write(6,fmt=fmt01)
      endif
      tests_passed = 0
      tests_failed = 0
      do i = 1, np
        if (expct(i) .eqv. reslt(i)) then
            tests_passed = tests_passed + 1
        else
            tests_failed = tests_failed + 1
            if (tests_failed .le. 100) then
              write(6,fmt=fmt03//fmt10) i, reslt(i), expct(i)
            end if
        endif
      enddo
      if (tests_failed .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      endif
      return
    end subroutine checkl4

    ! logical*8
    subroutine checkl8(reslt, expct, np, atoler, rtoler)
!dir$ ignore_tkr (r) reslt, expct
      logical*8, dimension(*) :: reslt
      logical*8, dimension(*) :: expct
      integer :: np
      logical*8, optional :: atoler, rtoler
      integer i, tests_passed, tests_failed
      if (present(atoler) .or. present(rtoler)) then
        write(6,fmt=fmt01)
      endif
      tests_passed = 0
      tests_failed = 0
      do i = 1, np
        if (expct(i) .eqv. reslt(i)) then
            tests_passed = tests_passed + 1
        else
            tests_failed = tests_failed + 1
            if (tests_failed .le. 100) then
              write(6,fmt=fmt03//fmt10) i, reslt(i), expct(i)
            end if
        endif
      enddo
      if (tests_failed .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      endif
      return
    end subroutine checkl8
  !
  ! real*4
    subroutine checkr4(reslt, expct, np, atoler, rtoler, ulptoler, ieee)
!dir$ ignore_tkr (r) reslt, expct
      real*4, dimension(*) :: reslt
      real*4, dimension(*) :: expct
      integer :: np
      real*4, optional :: atoler, rtoler, ulptoler
      logical, optional :: ieee
      integer i, tests_passed, tests_failed, tests_tolerated
      real*4   abserror, relerror
      logical ieee_on, anytolerated

      anytolerated = present(atoler) .or. present(rtoler) .or. present(ulptoler)
      ieee_on = .false.
      if (present(ieee)) ieee_on = ieee

      tests_passed = 0
      tests_failed = 0
      tests_tolerated = 0

      do i = 1, np
        if (ieee_on) then
          iri = ieeecheckcases(reslt(i), expct(i))
          if (iri.eq.1) then
            goto 100
          else if (iri.eq.2) then
            tests_passed = tests_passed + 1
            cycle
          end if
        end if

        if (expct(i) .eq. reslt(i)) then
            tests_passed = tests_passed + 1
            cycle
        end if

        abserror = abs(expct(i) - reslt(i))
        if (present(atoler)) then
          if (abserror .gt. abs(atoler)) goto 100
        end if

        if (present(rtoler)) then
          relerror = abserror / amax1(abs(expct(i)),ieee_next_after(0.0,1.0))
          if (relerror .gt. abs(rtoler)) goto 100
        end if

        if (present(ulptoler)) then
          if (ulperror(reslt(i),expct(i)) .gt. abs(ulptoler)) goto 100
        end if

        if (anytolerated) then  ! Some tolerances, so if here we've passed
            tests_passed = tests_passed + 1
            tests_tolerated = tests_tolerated + 1
            if (tests_tolerated .le. 100) then
              write(6,fmt=fmt02//fmt16) i, reslt(i),reslt(i), expct(i),expct(i)
            end if
            cycle
        end if

  100   tests_failed = tests_failed + 1   ! No tolerances, here we've failed
        if (tests_failed .le. 100) then
            write(6,fmt=fmt03//fmt16) i, reslt(i),reslt(i), expct(i),expct(i)
        end if
      end do

      if (tests_failed .eq. 0) then
         if (tests_tolerated .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
         else
            write(6,fmt=fmt05) np, tests_passed, tests_tolerated
         end if
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      end if
      return
    end subroutine checkr4
    ! real*8
    subroutine checkr8(reslt, expct, np, atoler, rtoler, ulptoler, ieee)
!dir$ ignore_tkr (r) reslt, expct
      real*8, dimension(*) :: reslt
      real*8, dimension(*) :: expct
      integer :: np
      real*8, optional :: atoler, rtoler, ulptoler
      logical, optional :: ieee
      integer i, tests_passed, tests_failed, tests_tolerated
      real*8   abserror, relerror
      logical ieee_on, anytolerated

      anytolerated = present(atoler) .or. present(rtoler) .or. present(ulptoler)
      ieee_on = .false.
      if (present(ieee)) ieee_on = ieee

      tests_passed = 0
      tests_failed = 0
      tests_tolerated = 0

      do i = 1, np
        if (ieee_on) then
          iri = ieeecheckcases(reslt(i), expct(i))
          if (iri.eq.1) then
            goto 100
          else if (iri.eq.2) then
            tests_passed = tests_passed + 1
            cycle
          end if
        end if

        if (expct(i) .eq. reslt(i)) then
            tests_passed = tests_passed + 1
            cycle
        end if

        abserror = dabs(expct(i) - reslt(i))
        if (present(atoler)) then
          if (abserror .gt. dabs(atoler)) goto 100
        end if

        if (present(rtoler)) then
          relerror = abserror/dmax1(dabs(expct(i)),ieee_next_after(0.0d0,1.0d0))
          if (relerror .gt. dabs(rtoler)) goto 100
        end if

        if (present(ulptoler)) then
          if (ulperror(reslt(i),expct(i)) .gt. dabs(ulptoler)) goto 100
        end if

        if (anytolerated) then  ! Some tolerances, so if here we've passed
            tests_passed = tests_passed + 1
            tests_tolerated = tests_tolerated + 1
            if (tests_tolerated .le. 100) then
              write(6,fmt=fmt02//fmt17) i, reslt(i),reslt(i), expct(i),expct(i)
            end if
            cycle
        end if
  100   tests_failed = tests_failed + 1   ! No tolerances, here we've failed
        if (tests_failed .le. 100) then
            write(6,fmt=fmt03//fmt17) i, reslt(i),reslt(i), expct(i),expct(i)
        end if
      enddo

      if (tests_failed .eq. 0) then
         if (tests_tolerated .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
         else
            write(6,fmt=fmt05) np, tests_passed, tests_tolerated
         endif
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      endif
      return
    end subroutine checkr8
#ifdef __flang_quadfp__
    !real16
    subroutine checkr16(reslt, expct, np, atoler, rtoler, ulptoler, ieee)
!dir$ ignore_tkr (r) reslt, expct
      real*16, dimension(*) :: reslt
      real*16, dimension(*) :: expct
      integer :: np
      real*16, optional :: atoler, rtoler, ulptoler
      logical, optional :: ieee
      integer i, tests_passed, tests_failed, tests_tolerated
      real*16   abserror, relerror
      logical ieee_on, anytolerated

      anytolerated = present(atoler) .or. present(rtoler) .or. present(ulptoler)
      ieee_on = .false.
      if (present(ieee)) ieee_on = ieee

      tests_passed = 0
      tests_failed = 0
      tests_tolerated = 0

      do i = 1, np
        if (ieee_on) then
          iri = ieeecheckcases(reslt(i), expct(i))
          if (iri.eq.1) then
            goto 100
          else if (iri.eq.2) then
            tests_passed = tests_passed + 1
            cycle
          end if
        end if

        if (expct(i) .eq. reslt(i)) then
            tests_passed = tests_passed + 1
            cycle
        end if

        abserror = abs(expct(i) - reslt(i))
        if (present(atoler)) then
          if (abserror .gt. abs(atoler)) goto 100
        end if

        if (present(rtoler)) then
          relerror = abserror / max(abs(expct(i)),ieee_next_after(0.0_16,1.0_16))
          if (relerror .gt. abs(rtoler)) goto 100
        end if

        if (present(ulptoler)) then
          if (ulperror(reslt(i),expct(i)) .gt. abs(ulptoler)) goto 100
        end if

        if (anytolerated) then  ! Some tolerances, so if here we've passed
            tests_passed = tests_passed + 1
            tests_tolerated = tests_tolerated + 1
            if (tests_tolerated .le. 100) then
              write(6,fmt=fmt02//fmt21) i, reslt(i),reslt(i), expct(i),expct(i)
            end if
            cycle
        end if

  100   tests_failed = tests_failed + 1   ! No tolerances, here we've failed
        if (tests_failed .le. 100) then
            write(6,fmt=fmt03//fmt21) i, reslt(i),reslt(i), expct(i),expct(i)
        end if
      end do

      if (tests_failed .eq. 0) then
         if (tests_tolerated .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
         else
            write(6,fmt=fmt05) np, tests_passed, tests_tolerated
         end if
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      end if
      return
    end subroutine checkr16
#endif
    subroutine checkc4(reslt, expct, np, atoler, rtoler, ulptoler, ieee)
!dir$ ignore_tkr (r) reslt, expct
      complex*8, dimension(*) :: reslt
      complex*8, dimension(*) :: expct
      integer :: np
      complex*8, optional :: atoler, rtoler, ulptoler
      logical, optional :: ieee
      integer i, tests_passed, tests_failed, tests_tolerated
      real*4     abserror, relerror
      real*4     rres, rexp, rx
      logical ieee_on, anytolerated

      anytolerated = present(atoler) .or. present(rtoler) .or. present(ulptoler)
      ieee_on = .false.
      if (present(ieee)) ieee_on = ieee

      tests_passed = 0
      tests_failed = 0
      tests_tolerated = 0

      do i = 1, np
        if (ieee_on) then
          rres = real(reslt(i))
          rexp = real(expct(i))
          irri = ieeecheckcases(rres, rexp)
          rres = imag(reslt(i))
          rexp = imag(expct(i))
          icri = ieeecheckcases(rres, rexp)
          if ((irri.eq.1) .or. (icri.eq.1)) then
            goto 100
          else if ((irri.eq.2) .and. (icri.eq.2)) then
            tests_passed = tests_passed + 1
            cycle
          end if
        end if

        if (expct(i) .eq. reslt(i)) then
            tests_passed = tests_passed + 1
            cycle
        end if

        abserror = cabs(expct(i) - reslt(i))
        if (present(atoler)) then
          if (abserror .gt. cabs(atoler)) goto 100
        end if

        if (present(rtoler)) then
          relerror = abserror / amax1(cabs(expct(i)),ieee_next_after(0.0,1.0))
          if (relerror .gt. cabs(rtoler)) goto 100
        end if

        if (present(ulptoler)) then
          rres = real(reslt(i))
          rexp = real(expct(i))
          if (ulperror(rres,rexp) .gt. real(ulptoler)) goto 100
          rres = imag(reslt(i))
          rexp = imag(expct(i))
          if (ulperror(rres,rexp) .gt. imag(ulptoler)) goto 100
        end if

        if (anytolerated) then  ! Some tolerances, so if here we've passed
            tests_passed = tests_passed + 1
            tests_tolerated = tests_tolerated + 1
            if (tests_tolerated .le. 100) then
              write(6,fmt=fmt02//fmt18) i, reslt(i),reslt(i), expct(i),expct(i)
            end if
            cycle
        end if

  100   tests_failed = tests_failed + 1   ! No tolerances, here we've failed
        if (tests_failed .le. 100) then
            write(6,fmt=fmt03//fmt18) i, reslt(i),reslt(i), expct(i),expct(i)
        end if
      enddo

      if (tests_failed .eq. 0) then
         if (tests_tolerated .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
         else
            write(6,fmt=fmt05) np, tests_passed, tests_tolerated
         endif
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      endif
      return
    end subroutine checkc4
    ! complex*16
    subroutine checkc8(reslt, expct, np, atoler, rtoler, ulptoler, ieee)
!dir$ ignore_tkr (r) reslt, expct
      complex*16, dimension(*) :: reslt
      complex*16, dimension(*) :: expct
      integer :: np
      complex*16, optional :: atoler, rtoler, ulptoler
      logical, optional :: ieee
      integer i, tests_passed, tests_failed, tests_tolerated
      real*8     abserror, relerror
      real*8     rres, rexp, rx
      logical ieee_on, anytolerated

      anytolerated = present(atoler) .or. present(rtoler) .or. present(ulptoler)
      ieee_on = .false.
      if (present(ieee)) ieee_on = ieee

      tests_passed = 0
      tests_failed = 0
      tests_tolerated = 0

      do i = 1, np
        if (ieee_on) then
          rres = dreal(reslt(i))
          rexp = dreal(expct(i))
          irri = ieeecheckcases(rres, rexp)
          rres = dimag(reslt(i))
          rexp = dimag(expct(i))
          icri = ieeecheckcases(rres, rexp)
          if ((irri.eq.1) .or. (icri.eq.1)) then
            goto 100
          else if ((irri.eq.2) .and. (icri.eq.2)) then
            tests_passed = tests_passed + 1
            cycle
          end if
        end if

        if (expct(i) .eq. reslt(i)) then
            tests_passed = tests_passed + 1
            cycle
        end if

        abserror = cdabs(expct(i) - reslt(i))
        if (present(atoler)) then
          if (abserror .gt. cdabs(atoler)) goto 100
        end if

        if (present(rtoler)) then
          relerror = abserror / dmax1(cdabs(expct(i)),ieee_next_after(0.0d0,1.0d0))
          if (relerror .gt. cdabs(rtoler)) goto 100
        end if

        if (present(ulptoler)) then
          rres = dreal(reslt(i))
          rexp = dreal(expct(i))
          if (ulperror(rres,rexp) .gt. dreal(ulptoler)) goto 100
          rres = dimag(reslt(i))
          rexp = dimag(expct(i))
          if (ulperror(rres,rexp) .gt. dimag(ulptoler)) goto 100
        end if

        if (anytolerated) then  ! Some tolerances, so if here we've passed
            tests_passed = tests_passed + 1
            tests_tolerated = tests_tolerated + 1
            if (tests_tolerated .le. 100) then
              write(6,fmt=fmt02//fmt19) i, reslt(i),reslt(i), expct(i),expct(i)
            end if
            cycle
        end if

  100   tests_failed = tests_failed + 1   ! No tolerances, here we've failed
        if (tests_failed .le. 100) then
            write(6,fmt=fmt03//fmt19) i, reslt(i),reslt(i), expct(i),expct(i)
        end if
      enddo

      if (tests_failed .eq. 0) then
         if (tests_tolerated .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
         else
            write(6,fmt=fmt05) np, tests_passed, tests_tolerated
         endif
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      endif
      return
    end subroutine checkc8
#ifdef __flang_quadfp__
    ! complex*32
    subroutine checkc16(reslt, expct, np, atoler, rtoler, ulptoler, ieee)
!dir$ ignore_tkr (r) reslt, expct
      complex*32, dimension(*) :: reslt
      complex*32, dimension(*) :: expct
      integer :: np
      complex*32, optional :: atoler, rtoler, ulptoler
      logical, optional :: ieee
      integer i, tests_passed, tests_failed, tests_tolerated
      real*16     abserror, relerror
      real*16     rres, rexp, rx
      logical ieee_on, anytolerated

      anytolerated = present(atoler) .or. present(rtoler) .or. present(ulptoler)
      ieee_on = .false.
      if (present(ieee)) ieee_on = ieee

      tests_passed = 0
      tests_failed = 0
      tests_tolerated = 0

      do i = 1, np
        if (ieee_on) then
          rres = qreal(reslt(i))
          rexp = qreal(expct(i))
          irri = ieeecheckcases(rres, rexp)
          rres = qimag(reslt(i))
          rexp = qimag(expct(i))
          icri = ieeecheckcases(rres, rexp)
          if ((irri.eq.1) .or. (icri.eq.1)) then
            goto 100
          else if ((irri.eq.2) .and. (icri.eq.2)) then
            tests_passed = tests_passed + 1
            cycle
          end if
        end if

        if (expct(i) .eq. reslt(i)) then
            tests_passed = tests_passed + 1
            cycle
        end if

        abserror = cqabs(expct(i) - reslt(i))
        if (present(atoler)) then
          if (abserror .gt. cqabs(atoler)) goto 100
        end if

        if (present(rtoler)) then
          relerror = abserror / qmax(cqabs(expct(i)),ieee_next_after(0.0_16,1.0_16))
          if (relerror .gt. cqabs(rtoler)) goto 100
        end if

        if (present(ulptoler)) then
          rres = qreal(reslt(i))
          rexp = qreal(expct(i))
          if (ulperror(rres,rexp) .gt. qreal(ulptoler)) goto 100
          rres = qimag(reslt(i))
          rexp = qimag(expct(i))
          if (ulperror(rres,rexp) .gt. qimag(ulptoler)) goto 100
        end if

        if (anytolerated) then  ! Some tolerances, so if here we've passed
            tests_passed = tests_passed + 1
            tests_tolerated = tests_tolerated + 1
            if (tests_tolerated .le. 100) then
              write(6,fmt=fmt02//fmt22) i, reslt(i),reslt(i), expct(i),expct(i)
            end if
            cycle
        end if

  100   tests_failed = tests_failed + 1   ! No tolerances, here we've failed
        if (tests_failed .le. 100) then
            write(6,fmt=fmt03//fmt22) i, reslt(i),reslt(i), expct(i),expct(i)
        end if
      enddo

      if (tests_failed .eq. 0) then
         if (tests_tolerated .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
         else
            write(6,fmt=fmt05) np, tests_passed, tests_tolerated
         endif
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      endif
      return
    end subroutine checkc16
#endif

    ! Now character*1
    subroutine checkc1(reslt, expct, np, atoler, rtoler)
!dir$ ignore_tkr (r) reslt, expct
      character*1, dimension(*) :: reslt
      character*1, dimension(*) :: expct
      integer :: np
      character*1, optional :: atoler, rtoler
      integer i, tests_passed, tests_failed
      if (present(atoler) .or. present(rtoler)) then
        write(6,fmt=fmt01)
      endif
      tests_passed = 0
      tests_failed = 0
      do i = 1, np
        if (expct(i) .eq. reslt(i)) then
            tests_passed = tests_passed + 1
        else
            tests_failed = tests_failed + 1
            if (tests_failed .le. 100) then
              write(6,fmt=fmt03//fmt11) i, ichar(reslt(i)),reslt(i),ichar(expct(i)),expct(i)
            end if
        endif
      enddo
      if (tests_failed .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      endif
      return
    end subroutine checkc1

    ! Now c_ptr
    subroutine checkcptr(reslt, expct, np)
      use iso_c_binding
      type(c_ptr) :: reslt, expct
      integer :: np
      integer*1, dimension(:), pointer :: ia1, ia2
      integer i, tests_passed, tests_failed

      call c_f_pointer(reslt, ia1, (/np/))
      call c_f_pointer(expct, ia2, (/np/))
      tests_passed = 0
      tests_failed = 0
      do i = 1, np
        if (ia2(i) .eq. ia1(i)) then
            tests_passed = tests_passed + 1
        else
            tests_failed = tests_failed + 1
            if (tests_failed .le. 100) then
              write(6,fmt=fmt03//fmt11) i, ia1(i),ia1(i), ia2(i),ia2(i)
            end if
        endif
      enddo
      if (tests_failed .eq. 0) then
            write(6,fmt=fmt04) np, tests_passed
      else
            write(6,fmt=fmt06) np, tests_passed, tests_failed
      endif
      return
    end subroutine checkcptr

    ! Now c_ptr2d
    subroutine checkcptr2d(reslt, expct, isz, icnt)
      use iso_c_binding
      type(c_ptr) :: reslt, expct
      integer ::isz, icnt
      integer*1, dimension(:,:), pointer :: ia1, ia2
      integer i, tests_passed, tests_failed

      call c_f_pointer(reslt, ia1, (/ isz, icnt /))
      call c_f_pointer(expct, ia2, (/ isz, icnt /))
      tests_passed = 0
      tests_failed = 0
      do j = 1, icnt
        do i = 1, isz
          if (ia2(i,j) .eq. ia1(i,j)) then
            if (i .eq. isz) tests_passed = tests_passed + 1
          else
            tests_failed = tests_failed + 1
            if (tests_failed .le. 100) then
              write(6,fmt=fmt07//fmt20) "c_ptr", j, i, ia1(i,j),ia2(i,j)
            end if
            exit
          endif
        end do
      enddo
      if (tests_failed .eq. 0) then
            write(6,fmt=fmt04) icnt, tests_passed
      else
            write(6,fmt=fmt06) icnt, tests_passed, tests_failed
      endif
      return
    end subroutine checkcptr2d

    ! Just a simple routine for checking derived types
    subroutine checkbytes(reslt, expct, name, isz, icnt)
!dir$ ignore_tkr reslt, expct
      integer*1, dimension(isz,*) :: reslt
      integer*1, dimension(isz,*) :: expct
      character*(*) :: name
      integer :: isz, icnt
      integer i, tests_passed, tests_failed
      tests_passed = 0
      tests_failed = 0
      do j = 1, icnt
        do i = 1, isz
          if (expct(i,j) .eq. reslt(i,j)) then
            if (i .eq. isz) tests_passed = tests_passed + 1
          else
            tests_failed = tests_failed + 1
            if (tests_failed .le. 100) then
              write(6,fmt=fmt07//fmt20) name, j, i, reslt(i,j),expct(i,j)
            end if
            exit
          endif
        end do
      end do
      if (tests_failed .eq. 0) then
            write(6,fmt=fmt04) icnt, tests_passed
      else
            write(6,fmt=fmt06) icnt, tests_passed, tests_failed
      endif
      return
    end subroutine checkbytes

    ! A more complicated routine for checking derived types
    subroutine checkdt(reslt, expct, name, lcnt, lsize, mcnt, msize, moffset)
!dir$ ignore_tkr reslt, expct
      integer*1, dimension(*) :: reslt
      integer*1, dimension(*) :: expct
      character*(*) :: name
      integer :: lcnt, lsize, mcnt, msize(mcnt), moffset(mcnt)

      integer i,j,k,ii,ij,ik, tests_passed, tests_failed
      tests_passed = 0
      tests_failed = 0
      ik = 0
      do k = 1, lcnt     ! For each array element in the reslt/expct array
        do j = 1, mcnt     ! For each member in a dt element
          ij = moffset(j)  ! This is the offset from beginning of the element
          isz = msize(j)   ! This is the size of the member in the element
          do i = 1, isz       ! Loop over each char in the member
            ii = ik+ij+i
            if (expct(ii) .eq. reslt(ii)) then
                if (i .eq. isz) tests_passed = tests_passed + 1
            else
                tests_failed = tests_failed + 1
                if (tests_failed .le. 100) then
                  write(6,fmt=fmt08//fmt20) name, k, j, i, reslt(ii),expct(ii)
                end if
                exit
            endif
          end do
        end do
        ik = ik + lsize  ! jump to the next element in the array
      end do
      if (tests_failed .eq. 0) then
            write(6,fmt=fmt04) lcnt*mcnt, tests_passed
      else
            write(6,fmt=fmt06) lcnt*mcnt, tests_passed, tests_failed
      endif
      return
    end subroutine checkdt

end module check_mod
