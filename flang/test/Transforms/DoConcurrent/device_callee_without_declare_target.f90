! Tests that, when `do concurrent` loops are mapped to the device, a warning is
! emitted for every procedure called from device code that is neither defined
! in the translation unit nor marked `declare target`, since no device version
! of such a procedure is emitted.

! RUN: split-file --leading-lines %s %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=52 -J%t %t/ops.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -fdo-concurrent-to-openmp=device -J%t %t/main.f90 -o /dev/null 2>&1 \
! RUN:   | FileCheck %s --implicit-check-not="not marked DECLARE TARGET"
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -fopenmp-is-target-device -fdo-concurrent-to-openmp=device -J%t \
! RUN:   %t/main.f90 -o /dev/null 2>&1 \
! RUN:   | FileCheck %s --implicit-check-not="not marked DECLARE TARGET"
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -fdo-concurrent-to-openmp=host -J%t %t/main.f90 -o /dev/null 2>&1 \
! RUN:   | FileCheck %s --allow-empty --check-prefix=HOST

! HOST-NOT: not marked DECLARE TARGET

!--- ops.f90
module ops
  implicit none
contains
  pure real function unmarked(x)
    real, intent(in) :: x
    unmarked = x
  end function

  pure real function marked(x)
    !$omp declare target
    real, intent(in) :: x
    marked = x
  end function
end module

!--- main.f90
module local
  implicit none
  interface
    pure real function ext(x)
      real, intent(in) :: x
    end function

    pure real function ext_marked(x)
      !$omp declare target
      real, intent(in) :: x
    end function

    pure real function ext_indirect(x)
      real, intent(in) :: x
    end function
  end interface
contains
  pure real function defined(x)
    real, intent(in) :: x
! CHECK-DAG: warning: loc("{{.*}}main.f90":[[@LINE+1]]:{{[0-9]+}}): 'ext_indirect' is called from device code generated for a DO CONCURRENT loop, but it is not defined in this file and is not marked DECLARE TARGET; add '!$omp declare target' to its definition to generate a device version of it
    defined = ext_indirect(x)
  end function
end module

program main
  use ops
  use local
  implicit none
  real :: a(10)
  integer :: i

  do concurrent (i = 1:10)
! CHECK-DAG: warning: loc("{{.*}}main.f90":[[@LINE+1]]:{{[0-9]+}}): 'unmarked' is called from device code generated for a DO CONCURRENT loop, but it is not defined in this file and is not marked DECLARE TARGET
    a(i) = unmarked(real(i))
    a(i) = a(i) + marked(real(i))
! CHECK-DAG: warning: loc("{{.*}}main.f90":[[@LINE+1]]:{{[0-9]+}}): 'ext' is called from device code generated for a DO CONCURRENT loop, but it is not defined in this file and is not marked DECLARE TARGET
    a(i) = a(i) + ext(real(i))
    a(i) = a(i) + ext_marked(real(i))
    a(i) = a(i) + defined(real(i))
    a(i) = a(i) + sin(real(i))
  end do
end program
