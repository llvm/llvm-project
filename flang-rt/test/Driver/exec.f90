! Verify that flang can correctly build executables.

! RUN: %flang -L"%libdir" %s %deplibs -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

! CHECK: Hello, World!
program hello
  print *, "Hello, World!"
end program
