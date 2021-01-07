! Check that lit configuration works by checking the compiler version

! VERSION-NOT:{{![[:space:]]}}
! VERSION:{{[[:space:]]}}
! VERSION-SAME:f18 compiler (under development), version {{[1-9][0-9]*.[0-9]*.[0-9]*}}
! VERSION-EMPTY:
  
! RUN: %f18 -V 2>&1 | FileCheck  -check-prefix=VERSION %s
! RUN: %f18 -v 2>&1 | FileCheck  -check-prefix=VERSION %s
! RUN: %f18 --version 2>&1 | FileCheck  -check-prefix=VERSION %s
