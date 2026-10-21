! RUN: split-file %s %t
! RUN: not %flang_fc1 -emit-llvm -O1 -fopenmp -fopenmp-version=52 \
! RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -o - %t/enter-folded.f90 \
! RUN:   2>&1 | FileCheck %s --check-prefix=ENTER
! RUN: not %flang_fc1 -emit-llvm -O1 -fopenmp -fopenmp-version=52 \
! RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -o - %t/enter-live.f90 \
! RUN:   2>&1 | FileCheck %s --check-prefix=ENTER
! RUN: not %flang_fc1 -emit-llvm -O1 -fopenmp -fopenmp-version=52 \
! RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -o - %t/exit-folded.f90 \
! RUN:   2>&1 | FileCheck %s --check-prefix=EXIT
! RUN: not %flang_fc1 -emit-llvm -O1 -fopenmp -fopenmp-version=52 \
! RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -o - %t/exit-live.f90 \
! RUN:   2>&1 | FileCheck %s --check-prefix=EXIT
! RUN: not %flang_fc1 -emit-llvm -O1 -fopenmp -fopenmp-version=52 \
! RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -o - %t/update-folded.f90 \
! RUN:   2>&1 | FileCheck %s --check-prefix=UPDATE
! RUN: not %flang_fc1 -emit-llvm -O1 -fopenmp -fopenmp-version=52 \
! RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -o - %t/update-live.f90 \
! RUN:   2>&1 | FileCheck %s --check-prefix=UPDATE

! Valid HLFIR dependences must be rejected at the translation boundary.
! ENTER: Unhandled clause depend in omp.target_enter_data operation
! EXIT: Unhandled clause depend in omp.target_exit_data operation
! UPDATE: Unhandled clause depend in omp.target_update operation

!--- enter-folded.f90
subroutine s(a)
  integer :: a(2)
  !$omp target enter data map(to:a) &
  !$omp& depend(iterator(i=1:1), in:a(1+0*i))
end

!--- enter-live.f90
subroutine s(a)
  integer :: a(2)
  !$omp target enter data map(to:a) &
  !$omp& depend(iterator(i=1:1), in:a(i))
end

!--- exit-folded.f90
subroutine s(a)
  integer :: a(2)
  !$omp target exit data map(from:a) &
  !$omp& depend(iterator(i=1:1), in:a(1+0*i))
end

!--- exit-live.f90
subroutine s(a)
  integer :: a(2)
  !$omp target exit data map(from:a) &
  !$omp& depend(iterator(i=1:1), in:a(i))
end

!--- update-folded.f90
subroutine s(a)
  integer :: a(2)
  !$omp target update to(a) &
  !$omp& depend(iterator(i=1:1), in:a(1+0*i))
end

!--- update-live.f90
subroutine s(a)
  integer :: a(2)
  !$omp target update to(a) &
  !$omp& depend(iterator(i=1:1), in:a(i))
end
