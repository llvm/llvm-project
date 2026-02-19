! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: compare
subroutine compare(x, c1, c2)
  character(len=4) c1, c2
  logical x
  ! CHECK: hlfir.cmpchar slt
  x = c1 < c2
end subroutine compare
