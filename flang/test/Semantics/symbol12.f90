! RUN: %python %S/test_symbols.py %s %flang_fc1
! Verify that SAVE attribute is propagated by EQUIVALENCE

!DEF: /s1 (Subroutine) Subprogram
subroutine s1
 !DEF: /s1/a SAVE ObjectEntity REAL(4)
 !DEF: /s1/b SAVE ObjectEntity REAL(4)
 !DEF: /s1/c SAVE ObjectEntity REAL(4)
 !DEF: /s1/d SAVE ObjectEntity REAL(4)
 real a, b, c, d
 !REF: /s1/d
 save :: d
 !REF: /s1/a
 !REF: /s1/b
 equivalence(a, b)
 !REF: /s1/b
 !REF: /s1/c
 equivalence(b, c)
 !REF: /s1/c
 !REF: /s1/d
 equivalence(c, d)
 !DEF: /s1/e ObjectEntity INTEGER(4)
 !DEF: /s1/f ObjectEntity INTEGER(4)
 equivalence(e, f)
 !REF: /s1/e
 !REF: /s1/f
 integer e, f
end subroutine
