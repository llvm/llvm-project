! Test that assumed length character scalars and explicit shape arrays are passed via
! CFI descriptor (fir.box) in BIND(C) procedures. They are passed only by address
! and length  in non BIND(C) procedures. See Fortran 2018 standard 18.3.6 point 2(5).
! RUN: bbc -hlfir -emit-fir -o - %s 2>&1 | FileCheck %s

module bindcchar
contains
! CHECK-LABEL: func.func @bindc(
! CHECK-SAME: %{{[^:]*}}: !fir.box<!fir.char<1,?>>
! CHECK-SAME: %{{[^:]*}}: !fir.box<!fir.array<100x!fir.char<1,?>>>
subroutine bindc(c1, c3) bind(c)
  character(*) ::  c1, c3(100)
 print *, c1(1:3), c3(5)(1:3)
end subroutine

! CHECK-LABEL:  func.func @bindc_optional(
! CHECK-SAME: %{{[^:]*}}: !fir.box<!fir.char<1,?>>
! CHECK-SAME: %{{[^:]*}}: !fir.box<!fir.array<100x!fir.char<1,?>>>
subroutine bindc_optional(c1, c3) bind(c)
  character(*), optional ::  c1, c3(100)
 print *, c1(1:3), c3(5)(1:3)
end subroutine

! CHECK-LABEL:  func.func @_QMbindccharPnot_bindc(
! CHECK-SAME: %{{[^:]*}}: !fir.boxchar<1>
! CHECK-SAME: %{{[^:]*}}: !fir.boxchar<1>
subroutine not_bindc(c1, c3)
  character(*) :: c1,  c3(100)
  call bindc(c1, c3)
  call bindc_optional(c1, c3)
end subroutine

! CHECK-LABEL:  func.func @_QMbindccharPnot_bindc_optional(
! CHECK-SAME: %{{[^:]*}}: !fir.boxchar<1>
! CHECK-SAME: %{{[^:]*}}: !fir.boxchar<1>
subroutine not_bindc_optional(c1, c3)
  character(*), optional :: c1,  c3(100)
  call bindc(c1, c3)
  call bindc_optional(c1, c3)
end subroutine
end module
