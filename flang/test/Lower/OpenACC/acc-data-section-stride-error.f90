! RUN: split-file %s %t
! RUN: bbc -fopenacc -emit-hlfir %t/omit-both.f90 -o - 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: bbc -fopenacc -emit-hlfir %t/omit-upper.f90 -o - 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: bbc -fopenacc -emit-hlfir %t/omit-lower.f90 -o - 2>&1 | FileCheck %s --check-prefix=ERROR

! ERROR: stride cannot be specified on an array section

!--- omit-both.f90
subroutine omit_both(a)
  real :: a(10)
  !$acc data copyin(a(::2))
  !$acc end data
end

!--- omit-upper.f90
subroutine omit_upper(a)
  real :: a(10)
  !$acc data copyin(a(2::2))
  !$acc end data
end

!--- omit-lower.f90
subroutine omit_lower(a)
  real :: a(10)
  !$acc data copyin(a(:8:2))
  !$acc end data
end
