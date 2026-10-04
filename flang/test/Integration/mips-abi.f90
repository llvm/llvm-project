!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! REQUIRES: mips-registered-target
! REQUIRES: module-independent
! RUN: %flang_fc1 -triple mips64-unknown-linux-gnuabi64 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,N64
! RUN: %flang_fc1 -triple mips64el-unknown-linux-gnuabi64 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,N64
! RUN: %flang_fc1 -triple mips64-unknown-linux-gnuabin32 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,N32
! RUN: %flang_fc1 -triple mips64el-unknown-linux-gnuabin32 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,N32

! N32: target datalayout = "{{.*}}-p:32:32-{{.*}}"
! N64: target datalayout = "{{[Ee]}}-m:e-i8:8:32-{{.*}}"

! CHECK-LABEL: define { float, float } @complex_value(float {{.*}}, float {{.*}})
function complex_value(z) result(r) bind(c)
  complex, value :: z
  complex :: r
  r = z
end function

! CHECK-LABEL: define signext i32 @integer_value(i32 signext {{.*}})
function integer_value(x) result(r) bind(c)
  integer, value :: x
  integer :: r
  r = x
end function

! N32-LABEL: define i64 @character_length_(ptr {{.*}}, i32 {{.*}})
! N64-LABEL: define i64 @character_length_(ptr {{.*}}, i64 {{.*}})
function character_length(s) result(n)
  character(*) :: s
  integer(8) :: n
  n = len(s, kind=8)
end function
