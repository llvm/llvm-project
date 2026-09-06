!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: bbc -outline-intrinsics %s -o - | tco --disable-llvm --mlir-print-ir-after=fir-to-llvm-ir 2>&1 | FileCheck %s

! Test properties of intrinsic function wrappers

! Test that intrinsic wrappers have internal linkage
function foo(x)
  foo = acos(x)
end function

! CHECK: llvm.func internal @fir.acos.contract.f32.f32
