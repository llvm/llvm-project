! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: compare
subroutine compare(x, c1, c2)
  character(len=4) c1, c2
  logical x
  ! CHECK: %[[RES:.*]] = fir.call @_FortranACharacterCompareScalar1
  ! CHECK: cmpi slt, %[[RES]],
  x = c1 < c2
end subroutine compare
