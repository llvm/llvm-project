! RUN: %python %S/test_errors.py %s %flang_fc1
! Test error messages for repeat specifier before control edit descriptors
! in FORMAT statements, WRITE format strings, named constant formats,
! and FMT= specifier.

  character(*), parameter :: fmt1 = "(2SS, F10.3)"

  ! Repeat specifier before sign-edit-desc in WRITE format strings
  !ERROR: Repeat specifier before 'SS' edit descriptor
  write(*,'(2SS, F10.3)') 0.5
  !ERROR: Repeat specifier before 'SP' edit descriptor
  write(*,'(2SP, F10.3)') 0.5
  !ERROR: Repeat specifier before 'S' edit descriptor
  write(*,'(2S, F10.3)') 0.5

  ! Repeat specifier before blank-interp-edit-desc in WRITE format strings
  !ERROR: Repeat specifier before 'BN' edit descriptor
  write(*,'(2BN, F10.3)') 0.5
  !ERROR: Repeat specifier before 'BZ' edit descriptor
  write(*,'(2BZ, F10.3)') 0.5

  ! Repeat specifier in named constant format
  !ERROR: Repeat specifier before 'SS' edit descriptor
  write(*,fmt1) 0.5

  ! Repeat specifier in FMT= specifier
  !ERROR: Repeat specifier before 'SS' edit descriptor
  write(*,fmt="(2SS, F10.3)") 0.5

  ! Repeat specifier before sign-edit-desc in FORMAT statements
  !ERROR: Repeat specifier before 'SS' edit descriptor
2001 format(2SS, F10.3)
  !ERROR: Repeat specifier before 'SP' edit descriptor
2002 format(2SP, F10.3)
  !ERROR: Repeat specifier before 'S' edit descriptor
2003 format(2S, F10.3)

  ! Repeat specifier before blank-interp-edit-desc in FORMAT statements
  !ERROR: Repeat specifier before 'BN' edit descriptor
2004 format(2BN, F10.3)
  !ERROR: Repeat specifier before 'BZ' edit descriptor
2005 format(2BZ, F10.3)

  ! Repeat specifier before round-edit-desc in FORMAT statements
  !ERROR: Repeat specifier before 'RU' edit descriptor
2006 format(2RU, F10.3)
  !ERROR: Repeat specifier before 'RZ' edit descriptor
2007 format(2RZ, F10.3)

  ! Repeat specifier before decimal-edit-desc in FORMAT statements
  !ERROR: Repeat specifier before 'DC' edit descriptor
2008 format(2DC, F10.3)
  !ERROR: Repeat specifier before 'DP' edit descriptor
2009 format(2DP, F10.3)

  end
