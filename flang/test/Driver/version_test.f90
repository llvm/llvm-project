! Check that lit configuration works by checking the compiler version

! RUN: %f18 -V 2>&1 | FileCheck  -check-prefix=VERSION %s
! VERSION-NOT:{{![[:space:]]}}
! VERSION:{{[[:space:]]}}
! VERSION-SAME:f18 compiler (under development)
! VERSION-EMPTY:
