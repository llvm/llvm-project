!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -emit-llvm -O0 %s -o - \
! RUN:   | FileCheck %s --check-prefixes=CHECK,INT-INLINE,FP-INLINE
! RUN: %flang_fc1 -emit-llvm -O1 %s -o - \
! RUN:   | FileCheck %s --check-prefixes=CHECK,INT-INLINE,FP-INLINE
! RUN: %flang_fc1 -emit-llvm -O2 %s -o - \
! RUN:   | FileCheck %s --check-prefixes=CHECK,INT-INLINE,FP-INLINE
! RUN: %flang_fc1 -emit-llvm -O3 %s -o - \
! RUN:   | FileCheck %s --check-prefixes=CHECK,INT-INLINE,FP-INLINE
! RUN: %flang_fc1 -emit-llvm -O0 -fcompensated-sum %s -o - \
! RUN:   | FileCheck %s --check-prefixes=CHECK,INT-INLINE,FP-RUNTIME
! RUN: %flang_fc1 -emit-llvm -O1 -fcompensated-sum %s -o - \
! RUN:   | FileCheck %s --check-prefixes=CHECK,INT-INLINE,FP-RUNTIME
! RUN: %flang_fc1 -emit-llvm -O2 -fcompensated-sum %s -o - \
! RUN:   | FileCheck %s --check-prefixes=CHECK,INT-INLINE,FP-RUNTIME
! RUN: %flang_fc1 -emit-llvm -O3 -fcompensated-sum %s -o - \
! RUN:   | FileCheck %s --check-prefixes=CHECK,INT-INLINE,FP-RUNTIME

integer function sum_integer(x)
  integer, intent(in) :: x(10)
  sum_integer = sum(x)
end function

! CHECK-LABEL: define {{.*}}@sum_integer_(
! INT-INLINE-NOT: @_FortranASumInteger4(
! INT-INLINE: add {{.*}}i32
! INT-INLINE-NOT: @_FortranASumInteger4(
! CHECK: ret i32

real function sum_real(x)
  real, intent(in) :: x(10)
  sum_real = sum(x)
end function

! CHECK-LABEL: define {{.*}}@sum_real_(
! FP-INLINE-NOT: @_FortranASumReal4(
! FP-INLINE: fadd{{.*}}float
! FP-INLINE-NOT: @_FortranASumReal4(
! FP-RUNTIME: call {{.*}}@_FortranASumReal4(
! CHECK: ret float

complex function sum_complex(x)
  complex, intent(in) :: x(10)
  sum_complex = sum(x)
end function

! CHECK-LABEL: define {{.*}}@sum_complex_(
! FP-INLINE-NOT: @_FortranACppSumComplex4(
! FP-INLINE: fadd{{.*}}float
! FP-INLINE-NOT: @_FortranACppSumComplex4(
! FP-RUNTIME: call void @_FortranACppSumComplex4(
! CHECK: ret
