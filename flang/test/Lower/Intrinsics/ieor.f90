! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: ieor_test
subroutine ieor_test(a, b)
  integer :: a, b
  print *, ieor(a, b)
  ! CHECK: %{{[0-9]+}} = arith.xori %{{[0-9]+}}, %{{[0-9]+}} : i{{(8|16|32|64|128)}}
end subroutine ieor_test

