! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: conjg_test
subroutine conjg_test(z1, z2)
  complex :: z1, z2
  ! CHECK: fir.extract_value
  ! CHECK: negf
  ! CHECK: fir.insert_value
  z2 = conjg(z1)
end subroutine
