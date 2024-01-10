! Verify that flang can correctly build executables.

! RUN: %flang %s -o %t
! RUN: %t | FileCheck %s
! RUN: rm -f %t

! CHECK: Hello, World!
program hello
  print *, "Hello, World!"
end program
