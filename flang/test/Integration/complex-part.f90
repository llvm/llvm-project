!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: bbc %s -o - | tco | FileCheck %s
! RUN: %flang -emit-llvm -S -mmlir -disable-external-name-interop %s -o - | FileCheck %s

  COMPLEX c
  c%RE = 3.14
  CALL sub(c)
END

! Verify that the offset in the struct does not regress from i32.
! CHECK-LABEL: define void @_QQmain()
! CHECK: getelementptr { float, float }, ptr %{{[0-9]+}}, i32 0, i32 0
