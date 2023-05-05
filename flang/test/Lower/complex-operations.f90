! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: @_QPadd_test
subroutine add_test(a,b,c)
  complex :: a, b, c
  ! CHECK-NOT: fir.extract_value
  ! CHECK-NOT: fir.insert_value
  ! CHECK: fir.addc {{.*}}: !fir.complex
  a = b + c
end subroutine add_test

! CHECK-LABEL: @_QPsub_test
subroutine sub_test(a,b,c)
  complex :: a, b, c
  ! CHECK-NOT: fir.extract_value
  ! CHECK-NOT: fir.insert_value
  ! CHECK: fir.subc {{.*}}: !fir.complex
  a = b - c
end subroutine sub_test

! CHECK-LABEL: @_QPmul_test
subroutine mul_test(a,b,c)
  complex :: a, b, c
  ! CHECK-NOT: fir.extract_value
  ! CHECK-NOT: fir.insert_value
  ! CHECK: fir.mulc {{.*}}: !fir.complex
  a = b * c
end subroutine mul_test

! CHECK-LABEL: @_QPdiv_test
subroutine div_test(a,b,c)
  complex :: a, b, c
  ! CHECK-NOT: fir.extract_value
  ! CHECK-NOT: fir.insert_value
  ! CHECK: fir.divc {{.*}}: !fir.complex
  a = b / c
end subroutine div_test
