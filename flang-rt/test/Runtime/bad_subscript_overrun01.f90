! Test a buffer overrun when an illegal vector-valued subscript is used in
! namelist input.
! The error should be reported through the READ's IOSTAT value.
! RUN: %flang -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s
! CHECK-NOT: FAIL

PROGRAM bad_subscript_overrun01
  IMPLICIT NONE

  INTEGER,DIMENSION(5)                :: U, EXPECTU
  INTEGER,DIMENSION(5)                :: V, EXPECTV
  CHARACTER(LEN=20)                   :: NAMELIST_RECORD
  INTEGER                             :: IOS, ILOOP

  NAMELIST /SCORES/ U
  NAMELIST /CHTEST/ NAMELIST_RECORD

  U = 8
  EXPECTU = U
  V = 9
  EXPECTV = V
  IOS = 0

  NAMELIST_RECORD = " &SCORES U(V) = -1 /"
  OPEN(UNIT=10, ACCESS='SEQUENTIAL', ACTION='READWRITE')
  WRITE(10,'(A)') NAMELIST_RECORD
  REWIND 10

  ! This should fail because of the bad index.
  READ(10, NML = SCORES, IOSTAT=IOS)
  CLOSE(UNIT=10, STATUS='KEEP')

  ! Make sure the READ status was a failure.
  IF(IOS.NE.0) THEN
    PRINT *, "Test 1 - PASS"
  ELSE
    print *, "Test 1 - FAIL"
  ENDIF

  ! Make sure the values of the array haven't changed.
  DO ILOOP = 1,5
    IF(U(ILOOP).EQ.EXPECTU(ILOOP)) THEN
      PRINT *, "Test 2 - PASS"
    ELSE
      PRINT *, "Test 2 - FAIL"
    ENDIF
  ENDDO

END PROGRAM bad_subscript_overrun01
