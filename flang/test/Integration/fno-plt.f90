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

! RUN: %flang_fc1 -emit-llvm -fno-plt -o %t.no-plt.ll %s
! RUN: grep -qE 'define void @test_\(.*\) #[0-9]+ {' %t.no-plt.ll || FileCheck <%t.no-plt.ll %s --check-prefix=NOPLT,NOPLT-NO-TESTATTR
! RUN: grep -qE 'define void @test_\(.*\) {' %t.no-plt.ll || FileCheck <%t.no-plt.ll %s --check-prefix=NOPLT,NOPLT-TESTATTR
! RUN: %flang_fc1 -emit-llvm -o - %s | FileCheck %s --check-prefix=PLT

subroutine test(x)
  real :: x
  call ext
  x = sin(x)
  print *, x
end subroutine

! The '{' on the define line rejects nonlazybind on @test_ itself.
! NOPLT-NO-TESTATTR: define void @test_({{.*}}) {
! NOPLT-TESTATTR: define void @test_({{.*}}) #[[TESTATTR:[0-9]+]] {
! NOPLT: declare void @ext_() #[[EXTATTR:[0-9]+]]
! NOPLT: declare ptr @_FortranAioBeginExternalListOutput({{.*}}) #[[EXTATTR]]
! NOPLT: declare float @llvm.sin.f32(float) #[[INTR:[0-9]+]]
! NOPLT-TESTATTR-NOT: attributes #[[TESTATTR]] = {{.*}}nonlazybind
! NOPLT: attributes #[[EXTATTR]] = { {{.*}}nonlazybind
! NOPLT-NOT: attributes #[[INTR]] = {{.*}}nonlazybind
! NOPLT: !{i32 7, !"RtLibUseGOT", i32 1}
! PLT-NOT: nonlazybind
! PLT-NOT: RtLibUseGOT
