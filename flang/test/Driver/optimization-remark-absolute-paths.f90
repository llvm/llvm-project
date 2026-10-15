! Test the behavior of -fdiagnostics-absolute-paths for optimization remarks:
! when enabled, flang prints the absolute path of the file the remark refers to
! instead of the (relative) path it was invoked with.

! REQUIRES: system-linux

! RUN: rm -rf %t.dir && mkdir -p %t.dir
! RUN: cp %s %t.dir/optimization-remark-absolute-paths.f90
! RUN: cd %t.dir && %flang_fc1 -emit-obj -O2 -vectorize-loops -vectorize-slp \
! RUN:     -debug-info-kind=standalone -Rpass=loop-vectorize \
! RUN:     optimization-remark-absolute-paths.f90 -o /dev/null 2>&1 \
! RUN:   | FileCheck %s --check-prefix=RELATIVE
! RUN: cd %t.dir && %flang_fc1 -emit-obj -O2 -vectorize-loops -vectorize-slp \
! RUN:     -debug-info-kind=standalone -fdiagnostics-absolute-paths \
! RUN:     -Rpass=loop-vectorize \
! RUN:     optimization-remark-absolute-paths.f90 -o /dev/null 2>&1 \
! RUN:   | FileCheck %s --check-prefix=ABSOLUTE

! RELATIVE: {{^}}optimization-remark-absolute-paths.f90:{{[0-9]+}}:{{[0-9]+}}: remark: vectorized loop
! ABSOLUTE: {{^}}/{{.*}}optimization-remark-absolute-paths.f90:{{[0-9]+}}:{{[0-9]+}}: remark: vectorized loop

subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)
  integer :: i
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine saxpy
