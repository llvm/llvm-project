!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! nonlazybind and the RtLibUseGOT module flag are applied to the LLVM module
! after FIR is translated, so -emit-hlfir and -emit-fir cannot observe them.
! -fno-plt marks an external procedure and a compiler runtime call nonlazybind
! and requests GOT for compiler-rt. A procedure defined in this file, and an
! intrinsic, are left unchanged. The default does neither.

! RUN: %flang_fc1 -emit-llvm -fno-plt -o - %s | FileCheck %s --check-prefix=NOPLT
! RUN: %flang_fc1 -emit-llvm -o - %s | FileCheck %s --check-prefix=PLT

subroutine test(x)
  real :: x
  call ext
  x = sin(x)
  print *, x
end subroutine

! The '{' on the define line rejects nonlazybind on @test_ itself.
! NOPLT: define void @test_({{.*}}) {
! NOPLT: declare void @ext_() #[[ATTR:[0-9]+]]
! NOPLT: declare ptr @_FortranAioBeginExternalListOutput({{.*}}) #[[ATTR]]
! NOPLT: declare float @llvm.sin.f32(float) #[[INTR:[0-9]+]]
! NOPLT: attributes #[[ATTR]] = { nonlazybind }
! NOPLT-NOT: attributes #[[INTR]] = {{.*}}nonlazybind
! NOPLT: !"RtLibUseGOT"
! PLT-NOT: nonlazybind
! PLT-NOT: RtLibUseGOT
