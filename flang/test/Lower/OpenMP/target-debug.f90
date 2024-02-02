!RUN: %flang_fc1 -triple amdgcn-amd-amdhsa %s -debug-info-kind=line-tables-only -fopenmp -fopenmp-is-target-device -emit-llvm -o - | FileCheck %s
program test
implicit none

  integer(kind = 4) :: a, b, c, d

  !$omp target map(tofrom: a, b, c, d)
  a = a + 1
  ! CHECK: !DILocation(line: [[@LINE-1]]
  b = a + 2
  ! CHECK: !DILocation(line: [[@LINE-1]]
  c = a + 3
  ! CHECK: !DILocation(line: [[@LINE-1]]
  d = a + 4
  ! CHECK: !DILocation(line: [[@LINE-1]]
  !$omp end target

end program test
