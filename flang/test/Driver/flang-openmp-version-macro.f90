! Test predefined _OPENMP macro which denotes OpenMP version

! RUN: %flang_fc1 -fopenmp -cpp -E %s | FileCheck %s --check-prefix=DEFAULT-OPENMP-VERSION
! RUN: %flang_fc1 -fopenmp -fopenmp-version=11 -cpp -E %s | FileCheck %s --check-prefix=OPENMP-VERSION-11
! RUN: %flang_fc1 -fopenmp -fopenmp-version=11 -cpp -E %s | FileCheck %s --check-prefix=OPENMP-VERSION-11
! RUN: %flang_fc1 -fopenmp -fopenmp-version=20 -cpp -E %s | FileCheck %s --check-prefix=OPENMP-VERSION-20
! RUN: %flang_fc1 -fopenmp -fopenmp-version=25 -cpp -E %s | FileCheck %s --check-prefix=OPENMP-VERSION-25
! RUN: %flang_fc1 -fopenmp -fopenmp-version=30 -cpp -E %s | FileCheck %s --check-prefix=OPENMP-VERSION-30
! RUN: %flang_fc1 -fopenmp -fopenmp-version=31 -cpp -E %s | FileCheck %s --check-prefix=OPENMP-VERSION-31
! RUN: %flang_fc1 -fopenmp -fopenmp-version=40 -cpp -E %s | FileCheck %s --check-prefix=OPENMP-VERSION-40
! RUN: %flang_fc1 -fopenmp -fopenmp-version=45 -cpp -E %s | FileCheck %s --check-prefix=OPENMP-VERSION-45
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -cpp -E %s | FileCheck %s --check-prefix=OPENMP-VERSION-50
! RUN: %flang_fc1 -fopenmp -fopenmp-version=51 -cpp -E %s | FileCheck %s --check-prefix=OPENMP-VERSION-51
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -cpp -E %s | FileCheck %s --check-prefix=OPENMP-VERSION-52

! DEFAULT-OPENMP-VERSION: integer :: var1 = 199911
! OPENMP-VERSION-11: integer :: var1 = 199911
! OPENMP-VERSION-20: integer :: var1 = 200011
! OPENMP-VERSION-25: integer :: var1 = 200505
! OPENMP-VERSION-30: integer :: var1 = 200805
! OPENMP-VERSION-31: integer :: var1 = 201107
! OPENMP-VERSION-40: integer :: var1 = 201307
! OPENMP-VERSION-45: integer :: var1 = 201511
! OPENMP-VERSION-50: integer :: var1 = 201811
! OPENMP-VERSION-51: integer :: var1 = 202011
! OPENMP-VERSION-52: integer :: var1 = 202111

#if _OPENMP
  integer :: var1 = _OPENMP
#endif
end program

