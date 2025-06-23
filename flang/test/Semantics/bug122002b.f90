! RUN: %python %S/test_errors.py %s %flang_fc1
SUBROUTINE sub00(a,b,n,m)
  complex(2) n,m
! ERROR: Must have INTEGER type, but is COMPLEX(2)
! ERROR: Must have INTEGER type, but is COMPLEX(2)
! ERROR: The type of 'b' has already been implicitly declared
  complex(3) a(n,m), b(size((LOG ((x * (a) - a + b / a - a))+1 - x)))
  a = a ** n
! ERROR: DO controls should be INTEGER
  DO 10 j = 1,m
    a = n ** a
    10   PRINT *, g
END SUBROUTINE sub00
