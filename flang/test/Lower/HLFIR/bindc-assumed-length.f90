! Test that assumed length character scalars and explicit shape arrays are passed via
! CFI descriptor (fir.box) in BIND(C) procedures. They are passed only by address
! and length  in non BIND(C) procedures. See Fortran 2018 standard 18.3.6 point 2(5).
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

! CHECK: func.func @foo(
! CHECK-SAME: %{{[^:]*}}: !fir.box<!fir.char<1,?>>
! CHECK-SAME: %{{[^:]*}}: !fir.box<!fir.array<100x!fir.char<1,?>>>
subroutine foo(c1, c3) bind(c)
  character(*) :: c1,  c3(100)
end subroutine

! CHECK: func.func @_QPnot_bindc(
! CHECK-SAME: %{{[^:]*}}: !fir.boxchar<1>
! CHECK-SAME: %{{[^:]*}}: !fir.boxchar<1>
subroutine not_bindc(c1, c3)
  character(*) :: c1,  c3(100)
end subroutine
