!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! REQUIRES: nvptx-registered-target

! RUN: %flang_fc1 -triple nvptx64-nvidia-cuda -fopenmp -fopenmp-is-target-device -emit-llvm %s -o %t

! Emit LLVM IR to exercise target ABI rewriting before function filtering.
! NVPTX does not implement complex argument or return marshalling, but this
! host-only routine is removed from the device module after lowering.
subroutine host_complex(x, y)
  complex(8) :: x, y
  y = sqrt(x) + exp(x)
  print *, abs(x)
end subroutine
