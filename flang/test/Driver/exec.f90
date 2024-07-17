! UNSUPPORTED: system-windows
! Verify that flang can correctly build executables.

! RUN: %flang %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%llvmshlibdir" %t | FileCheck %s
! RUN: rm -f %t

! CHECK: Hello, World!
program hello
  print *, "Hello, World!"
end program
