! RUN: bbc --strict-fir-volatile-verifier %s -o - | FileCheck %s

! Requires correct propagation of volatility for allocatable nested types.
! XFAIL: *

function allocatable_udt()
  type :: base_type
    integer :: i = 42
  end type
  type, extends(base_type) :: ext_type
    integer :: j = 100
  end type
  integer :: allocatable_udt
  type(ext_type), allocatable, volatile :: v2(:,:)
  allocate(v2(2,3))
  allocatable_udt = v2(1,1)%i
end function
