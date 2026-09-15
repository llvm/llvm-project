!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -emit-llvm %s -triple ppc64le-unknown-linux -o - | FileCheck %s
! REQUIRES: target=powerpc{{.*}}

! The shift amount of vec_sld must reach PowerPC lowering as a constant even
! when it is written as an element of a named constant (which the front end
! may keep in designator form for storage association).
subroutine test_sld_parameter(arg1, arg2, r)
  vector(integer(4)) :: arg1, arg2, r
  integer, parameter :: sh(1) = [2]
  r = vec_sld(arg1, arg2, sh(1))
end subroutine
! CHECK-LABEL: @test_sld_parameter_
! CHECK: shufflevector <16 x i8> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i32>
