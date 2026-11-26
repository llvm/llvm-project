! UNSUPPORTED: offload-cuda

! Verify that flang can correctly build executables.

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: %t | FileCheck %s

! CHECK: Hello, World!
program hello
  print *, "Hello, World!"
end program
