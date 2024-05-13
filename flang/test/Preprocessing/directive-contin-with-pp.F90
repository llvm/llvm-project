! RUN: %flang -E %s 2>&1 | FileCheck %s

#define DIR_START !dir$
#define DIR_CONT !dir$&
#define FIRST(x) DIR_START x
#define NEXT(x) DIR_CONT x
#define AMPER &

subroutine s(x1, x2, x3, x4, x5, x6, x7)

!dir$ ignore_tkr x1

!dir$ ignore_tkr &
!dir$& x2

DIR_START ignore_tkr x3

!dir$ ignore_tkr AMPER
DIR_CONT x4

FIRST(ignore_tkr &)
!dir$& x5

FIRST(ignore_tkr &)
NEXT(x6)

FIRST(ignore_tkr &)
NEXT(x7 &)
NEXT(x8)

end

!CHECK: subroutine s(x1, x2, x3, x4, x5, x6, x7)
!CHECK: !dir$ ignore_tkr x1
!CHECK: !dir$ ignore_tkr x2
!CHECK: !dir$ ignore_tkr x3
!CHECK: !dir$ ignore_tkr  x4
!CHECK: !dir$ ignore_tkr  x5
!CHECK: !dir$ ignore_tkr  x6
!CHECK: !dir$ ignore_tkr  x7  x8
!CHECK: end
