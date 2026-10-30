! RUN: bbc -emit-fir %s -o - | FileCheck %s --check-prefix=DEFAULT
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s --check-prefix=DEFAULT
! RUN: %flang_fc1 -fbackslash -emit-fir %s -o - | FileCheck %s --check-prefix=ESCAPES

! Quoted strings in a numbered FORMAT must be decoded like CHARACTER literals.
! FIR prints each literal backslash as two backslashes.
subroutine format_backslash
  integer label
  write (*, 10)
  write (*, 20)
  print 30, 1.5
  write (*, 40)
  write (*, 50)
  assign 60 to label
  write (*, label)
  write (*, 70)
  write (*, "(1x, 'inline\path')")
10 format (1x, 'a\b')
20 format (1x, "two \\ backslashes")
30 format (1x, '\alpha = ', f4.1)
40 format ('it''s \here')
! Hollerith text does not use escapes, even when it contains quotes.
50 format (7H'a\\b'!, 'c\\d')
60 format ('\assigned')
! Escapes that decode to quotes must not terminate the quoted string.
70 format ('\x27\047')
end

! DEFAULT-DAG: fir.string_lit "(1x, 'a\\b')"(11)
! ESCAPES-DAG: fir.string_lit "(1x, 'a\08')"(10)
! DEFAULT-DAG: fir.string_lit "(1x, \22two \\\\ backslashes\22)"(26)
! ESCAPES-DAG: fir.string_lit "(1x, \22two \\ backslashes\22)"(25)
! DEFAULT-DAG: fir.string_lit "(1x, '\\alpha = ', f4.1)"(23)
! ESCAPES-DAG: fir.string_lit "(1x, 'alpha = ', f4.1)"(22)
! DEFAULT-DAG: fir.string_lit "('it''s \\here')"(15)
! ESCAPES-DAG: fir.string_lit "('it''s here')"(14)
! DEFAULT-DAG: fir.string_lit "(7h'a\\\\b'!, 'c\\\\d')"(19)
! ESCAPES-DAG: fir.string_lit "(7h'a\\\\b'!, 'c\\d')"(18)
! DEFAULT-DAG: fir.string_lit "('\\assigned')"(13)
! ESCAPES-DAG: fir.string_lit "('assigned')"(12)
! DEFAULT-DAG: fir.string_lit "('\\x27\\047')"(12)
! ESCAPES-DAG: fir.string_lit "('''''')"(8)
! DEFAULT-DAG: fir.string_lit "(1x, 'inline\\path')"(19)
! ESCAPES-DAG: fir.string_lit "(1x, 'inlinepath')"(18)

! The prescanner and FORMAT grammar can disagree about Hollerith boundaries.
! In default mode remove only the backslashes the prescanner actually inserted.
subroutine format_hollerith_boundaries
  write (*, 10)
  write (*, 20)
  write (*, 30)
10 format (1 2hab'xxxxxxxxx, 'zz,3h\nb,1h')
20 format (x2h'a', 3h\nb, 1h')
! This cooks to the same text as the macro case in format-backslash-macros.F90,
! but the backslash before n was doubled by the prescanner in this case.
30 format (12h'abxxxxxxxxx, 'zz,3h\nb,1h')
end

! DEFAULT-DAG: fir.string_lit "(1 2hab'xxxxxxxxx, 'zz,3h\\nb,1h')"(33)
! ESCAPES-DAG: fir.string_lit "(1 2hab'xxxxxxxxx, 'zz,3h\0Ab,1h')"(32)
! DEFAULT-DAG: fir.string_lit "(x2h'a', 3h\\nb, 1h')"(20)
! ESCAPES-DAG: fir.string_lit "(x2h'a', 3h\0Ab, 1h')"(19)
! DEFAULT-DAG: fir.string_lit "(12h'abxxxxxxxxx, 'zz,3h\\nb,1h')"(32)
! ESCAPES-DAG: fir.string_lit "(12h'abxxxxxxxxx, 'zz,3h\0Ab,1h')"(31)
