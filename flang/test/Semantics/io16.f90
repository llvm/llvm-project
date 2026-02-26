! RUN: %python %S/test_errors.py %s %flang_fc1

! F202X leading-zero control edit descriptors: LZ, LZS, LZP

  ! Valid uses of LZ, LZP, LZS in FORMAT statements
1001 format(LZ, F10.3)
1002 format(LZP, F10.3)
1003 format(LZS, F10.3)
1004 format(LZ, E10.3)
1005 format(LZP, E10.3)
1006 format(LZS, E10.3)
1007 format(LZS, D10.3)
1008 format(LZ, G10.3)

  ! Valid uses with blanks inside keywords (Fortran ignores blanks)
1009 format(L Z, F10.3)
1010 format(L Z P, F10.3)
1011 format(L Z S, F10.3)

  ! Combining with other control edit descriptors
1012 format(LZP, DC, F10.3)
1013 format(BN, LZS, F10.3)
1014 format(LZ, SS, RZ, F10.3)

  ! Multiple groups
1015 format(LZP, 3F10.3, LZS, 2E12.4)

  ! C1302 : multiple edit descriptors without ',' separation; no errors
1016 format(LZF10.3)
1017 format(LZPF10.3)
1018 format(LZSF10.3)
1019 format(LZE10.3)
1020 format(LZPE10.3)
1021 format(LZSD10.3)
1022 format(LZG10.3)
1023 format(LZPDCF10.3)
1024 format(BNLZSF10.3)
1025 format(LZPF10.3LZSF10.3)
1026 format(LZP3F10.3LZS2E12.4)

  ! In WRITE format strings
  write(*, '(LZ, F10.3)') 0.5
  write(*, '(LZP, F10.3)') 0.5
  write(*, '(LZS, F10.3)') 0.5
  write(*, '(LZP,E10.3)') 0.5
  write(*, '(LZS,D10.3)') 0.5

  ! C1302 : WRITE format strings without ',' separation; no errors
  write(*, '(LZF10.3)') 0.5
  write(*, '(LZPF10.3)') 0.5
  write(*, '(LZSF10.3)') 0.5
  write(*, '(LZPE10.3)') 0.5
  write(*, '(LZP3F10.3LZS2E12.4)') 0.5, 0.5, 0.5, 0.5, 0.5

  ! FMT= specifier with comma-separated descriptors
  write(*, fmt='(LZ, F10.3)') 0.5
  write(*, fmt='(LZP, F10.3)') 0.5
  write(*, fmt='(LZS, F10.3)') 0.5
  write(*, fmt='(LZP, E10.3)') 0.5
  write(*, fmt='(LZS, D10.3)') 0.5
  write(*, fmt='(LZP, DC, F10.3)') 0.5
  write(*, fmt='(BN, LZS, F10.3)') 0.5

  ! FMT= specifier without ',' separation; no errors
  write(*, fmt='(LZF10.3)') 0.5
  write(*, fmt='(LZPF10.3)') 0.5
  write(*, fmt='(LZSF10.3)') 0.5
  write(*, fmt='(LZPE10.3)') 0.5
  write(*, fmt='(LZP3F10.3LZS2E12.4)') 0.5, 0.5, 0.5, 0.5, 0.5

  ! FMT= specifier with FORMAT label reference
  write(*, fmt=1001) 0.5
  write(*, fmt=1002) 0.5
  write(*, fmt=1017) 0.5

  ! LZ/LZP/LZS coexisting with abbreviated L (no width) data edit descriptor
  write(*, '(LZP, F10.3, L)') 0.5, .true.
  write(*, '(LZS, F10.3, L)') 0.5, .true.

  ! Error: repeat specifier before LZ/LZP/LZS in WRITE format strings
  !ERROR: Repeat specifier before 'LZ' edit descriptor
  write(*, '(3LZ, F10.3)') 0.5

  !ERROR: Repeat specifier before 'LZP' edit descriptor
  write(*, '(2LZP, F10.3)') 0.5

  !ERROR: Repeat specifier before 'LZS' edit descriptor
  write(*, '(2LZS, F10.3)') 0.5

  ! Error: repeat specifier before LZ/LZP/LZS in FORMAT statements
  !ERROR: Repeat specifier before 'LZ' edit descriptor
2001 format(3LZ, F10.3)

  !ERROR: Repeat specifier before 'LZP' edit descriptor
2002 format(2LZP, F10.3)

  !ERROR: Repeat specifier before 'LZS' edit descriptor
2003 format(2LZS, F10.3)
end
