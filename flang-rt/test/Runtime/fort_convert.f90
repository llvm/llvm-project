! UNSUPPORTED: offload-cuda
! UNSUPPORTED: system-windows

! Verify endian conversion for unformatted stream I/O with and
! without runtime environment variable FORT_CONVERT_UNIT specified.
!
! Default units:
! OPEN(10, FORM="unformatted")
! OPEN(11, FORM="unformatted", CONVERT="native")
! OPEN(12, FORM="unformatted", CONVERT="big_endian")
! OPEN(13, FORM="unformatted", CONVERT="little_endian")
! OPEN(14, FORM="unformatted", CONVERT="swap")
!
! First test, FORT_CONVERT_UNIT not present.
! Second test: FORT_CONVERT_UNIT="swap:10-11;little_endian:12;big_endian:13;native:14"

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: %t | FileCheck %s
! RUN: env FORT_CONVERT_UNIT="swap:10-11;little_endian:12;big_endian:13;native:14" %t | FileCheck %s

! CHECK: PASS
module testmod
  implicit none
  character(len=1), dimension(4) :: c4
  integer(kind=4) :: i4 = 1
  integer(kind=4), dimension(2) :: data10 = [10, int(z'01cafe23')]
  integer(kind=4), dimension(2) :: data11 = [11, int(z'45beef67')]
  integer(kind=4), dimension(2) :: data12 = [12, int(z'98faceab')]
  integer(kind=4), dimension(2) :: data13 = [13, int(z'cdfeed01')]
  integer(kind=4), dimension(2) :: data14 = [14, int(z'02deaf03')]

  contains
  logical function isLittleEndian()
    c4 = transfer(i4, c4)
    isLittleEndian = ichar(c4(1)) == 1
  end function isLittleEndian

  logical function isBigEndian()
    isBigEndian = .not. isLittleEndian()
  end function isBigEndian

  logical function filecheck(unit, refdata, isNative) result(res)
    integer(kind=4) :: unit
    integer(kind=4), dimension(2) :: refdata
    logical :: isNative

    integer :: i
    integer :: ios
    integer(kind=4), dimension(4) :: filearr
    integer(kind=4), dimension(4) :: workarr
    character(len=1), dimension(4) :: charwork

    res = .true.

    read(unit, iostat=ios) filearr
    if (ios /= 0) then
      print*, 'ios=', ios
      stop 1
    end if

    workarr(1) = int(z'00000008')
    workarr(2:3) = refdata
    workarr(4) = int(z'00000008')

    ! If isNative == .true. file data should match in memory layout of refdata.
    ! If isNative == .false., refdata (kind=4) has to have endianness switched.
    if (.not. isNative) then
      ! swap endianness of input reference data
      do i = 1, 4
        charwork = transfer(workarr(i), charwork)
        charwork = charwork(size(charwork):1:-1)
        workarr(i)  = transfer(charwork, workarr(i))
      end do
    end if

    res = .not. any(filearr /= workarr)

    if (.not. res) then
      write(*,'(4("0x", z8.8:x))') filearr
      write(*,'(4("0x", z8.8:x))') workarr
    endif

  end function filecheck
end module testmod
program main
  use testmod
  implicit none
  integer :: ios
  integer :: i
  logical :: FORT_CONVERT_UNIT_present

  call get_environment_variable("FORT_CONVERT_UNIT", length=ios)
  FORT_CONVERT_UNIT_present = ios /= 0

  ! Some runtime debug statements.
  ! print*, 'FORT_CONVERT_UNIT_present=', FORT_CONVERT_UNIT_present
  ! if (isLittleEndian()) print *, 'Little Endian'

  open(10, iostat=ios, form="unformatted", access="sequential", status="unknown")
  if (ios /= 0) stop 10
  open(11, iostat=ios, form="unformatted", access="sequential", status="unknown", convert="native")
  if (ios /= 0) stop 11
  open(12, iostat=ios, form="unformatted", access="sequential", status="unknown", convert="big_endian")
  if (ios /= 0) stop 12
  open(13, iostat=ios, form="unformatted", access="sequential", status="unknown", convert="little_endian")
  if (ios /= 0) stop 13
  open(14, iostat=ios, form="unformatted", access="sequential", status="unknown", convert="swap")
  if (ios /= 0) stop 14

  write(10, iostat=ios) data10
  if (ios /= 0) stop 20
  write(11, iostat=ios) data11
  if (ios /= 0) stop 21
  write(12, iostat=ios) data12
  if (ios /= 0) stop 22
  write(13, iostat=ios) data13
  if (ios /= 0) stop 23
  write(14, iostat=ios) data14
  if (ios /= 0) stop 24

  do i = 10, 14
    close(i)
  end do

  open(20, iostat=ios, form="unformatted", access="stream", name="fort.10", status="old", convert="native")
  if (ios /= 0) stop 30
  open(21, iostat=ios, form="unformatted", access="stream", name="fort.11", status="old", convert="native")
  if (ios /= 0) stop 31
  open(22, iostat=ios, form="unformatted", access="stream", name="fort.12", status="old", convert="native")
  if (ios /= 0) stop 32
  open(23, iostat=ios, form="unformatted", access="stream", name="fort.13", status="old", convert="native")
  if (ios /= 0) stop 33
  open(24, iostat=ios, form="unformatted", access="stream", name="fort.14", status="old", convert="native")
  if (ios /= 0) stop 34

 if (FORT_CONVERT_UNIT_present) then
    if (.not. filecheck(20, data10, .false.)) stop 40
    if (.not. filecheck(21, data11, .false.)) stop 41
    if (.not. filecheck(22, data12, isLittleEndian())) stop 42
    if (.not. filecheck(23, data13, isBigEndian())) stop 43
    if (.not. filecheck(24, data14, .true.)) stop 44
  else
    if (.not. filecheck(20, data10, .true.)) stop 45
    if (.not. filecheck(21, data11, .true.)) stop 46
    if (.not. filecheck(22, data12, isBigEndian())) stop 47
    if (.not. filecheck(23, data13, isLittleEndian())) stop 48
    if (.not. filecheck(24, data14, .false.)) stop 49
  endif

  print *,'PASS'
end program main
