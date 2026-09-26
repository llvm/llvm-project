! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! REQUIRES: target=x86_64{{.*}}
! UNSUPPORTED: system-windows

! REAL(10) is the x87 extended format: 80 significant bits held in a 16-byte
! container, so its raw bits occupy 16 bytes rather than 10.  Constant
! initialization, TRANSFER and host-library folding all round-trip values
! through their raw bytes, so the number of bytes stored must not be
! conflated with the kind number.

subroutine data_init
  real(10) :: x
  complex(10) :: y
  data x/1.5_10/
  data y/(1.5_10, 2.5_10)/
  print *, x, y
end subroutine
! CHECK-LABEL: SUBROUTINE data_init
! CHECK: DATA x/1.5_10/
! CHECK: DATA y/(1.5_10,2.5_10)/

subroutine transfers
  ! A bit pattern of 1 reinterpreted as REAL(10) is the smallest subnormal,
  ! so the low-order bytes must survive the round trip.
  print *, transfer(1_8, 0.0_10)
  print *, transfer(1.5_10, 0.0_10)
  print *, transfer(1.5_10, 0_2, 8)
end subroutine
! CHECK-LABEL: SUBROUTINE transfers
! CHECK: PRINT *, {{.*}}e-4951_10
! CHECK: PRINT *, 1.5_10
! CHECK: PRINT *, [INTEGER(2)::0_2,0_2,0_2,-16384_2,16383_2,0_2,0_2,0_2]

subroutine host_folding
  ! Folding these casts to and from the host long double, whose size differs
  ! from the number of significant bytes.
  real(10), parameter :: s = sin(1.0_10)
  real(10), parameter :: e = exp(1.0_10)
  print *, s, e
end subroutine
! CHECK-LABEL: SUBROUTINE host_folding
! CHECK: PRINT *, 8.4{{[0-9]*}}e-1_10, 2.7{{[0-9]*}}_10
