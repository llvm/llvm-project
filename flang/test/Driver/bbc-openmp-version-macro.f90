! Test predefined _OPENMP macro which denotes OpenMP version

! RUN: bbc -fopenmp -o -  %s | FileCheck %s --check-prefix=DEFAULT-OPENMP-VERSION
! RUN: bbc -fopenmp -fopenmp-version=11 -o - %s | FileCheck %s --check-prefix=OPENMP-VERSION-11
! RUN: bbc -fopenmp -fopenmp-version=11 -o - %s | FileCheck %s --check-prefix=OPENMP-VERSION-11
! RUN: bbc -fopenmp -fopenmp-version=20 -o - %s | FileCheck %s --check-prefix=OPENMP-VERSION-20
! RUN: bbc -fopenmp -fopenmp-version=25 -o - %s | FileCheck %s --check-prefix=OPENMP-VERSION-25
! RUN: bbc -fopenmp -fopenmp-version=30 -o - %s | FileCheck %s --check-prefix=OPENMP-VERSION-30
! RUN: bbc -fopenmp -fopenmp-version=31 -o - %s | FileCheck %s --check-prefix=OPENMP-VERSION-31
! RUN: bbc -fopenmp -fopenmp-version=40 -o - %s | FileCheck %s --check-prefix=OPENMP-VERSION-40
! RUN: bbc -fopenmp -fopenmp-version=45 -o - %s | FileCheck %s --check-prefix=OPENMP-VERSION-45
! RUN: bbc -fopenmp -fopenmp-version=50 -o - %s | FileCheck %s --check-prefix=OPENMP-VERSION-50
! RUN: bbc -fopenmp -fopenmp-version=51 -o - %s | FileCheck %s --check-prefix=OPENMP-VERSION-51
! RUN: bbc -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s --check-prefix=OPENMP-VERSION-52

! DEFAULT-OPENMP-VERSION: {{.*}} = arith.constant 199911 : i32
! OPENMP-VERSION-11: {{.*}} = arith.constant 199911 : i32
! OPENMP-VERSION-20: {{.*}} = arith.constant 200011 : i32
! OPENMP-VERSION-25: {{.*}} = arith.constant 200505 : i32
! OPENMP-VERSION-30: {{.*}} = arith.constant 200805 : i32
! OPENMP-VERSION-31: {{.*}} = arith.constant 201107 : i32
! OPENMP-VERSION-40: {{.*}} = arith.constant 201307 : i32
! OPENMP-VERSION-45: {{.*}} = arith.constant 201511 : i32
! OPENMP-VERSION-50: {{.*}} = arith.constant 201811 : i32
! OPENMP-VERSION-51: {{.*}} = arith.constant 202011 : i32
! OPENMP-VERSION-52: {{.*}} = arith.constant 202111 : i32

#if _OPENMP
  integer :: var1 = _OPENMP
#endif
end program

