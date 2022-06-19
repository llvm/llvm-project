! RUN: %python %S/test_modfile.py %s %flang_fc1
! Test declarations with coarray-spec

! Different ways of declaring the same coarray.
module m1
  real :: a(1:5)[1:10,1:*]
  real, dimension(5) :: b[1:10,1:*]
  real, codimension[1:10,1:*] :: c(5)
  real, codimension[1:10,1:*], dimension(5) :: d
  codimension :: e[1:10,1:*]
  dimension :: e(5)
  real :: e
end
!Expect: m1.mod
!module m1
! real(4)::a(1_8:5_8)[1_8:10_8,1_8:*]
! real(4)::b(1_8:5_8)[1_8:10_8,1_8:*]
! real(4)::c(1_8:5_8)[1_8:10_8,1_8:*]
! real(4)::d(1_8:5_8)[1_8:10_8,1_8:*]
! real(4)::e(1_8:5_8)[1_8:10_8,1_8:*]
!end

! coarray-spec in codimension and target statements.
module m2
  codimension :: a[10,*], b[*]
  target :: c[10,*], d[*]
end
!Expect: m2.mod
!module m2
! real(4)::a[1_8:10_8,1_8:*]
! real(4)::b[1_8:*]
! real(4),target::c[1_8:10_8,1_8:*]
! real(4),target::d[1_8:*]
!end

! coarray-spec in components and with non-constants bounds
module m3
  type t
    real, allocatable :: c[:,:]
    complex, allocatable, codimension[:,:] :: d
  end type
  real, allocatable :: e[:,:,:]
contains
  subroutine s(a, b, n)
    integer(8) :: n
    real :: a[1:n,2:*]
    real, codimension[1:n,2:*] :: b
  end
end
!Expect: m3.mod
!module m3
! type::t
!  real(4),allocatable::c[:,:]
!  complex(4),allocatable::d[:,:]
! end type
! real(4),allocatable::e[:,:,:]
!contains
! subroutine s(a,b,n)
!  integer(8)::n
!  real(4)::a[1_8:n,2_8:*]
!  real(4)::b[1_8:n,2_8:*]
! end
!end

! coarray-spec in both attributes and entity-decl
module m4
  real, codimension[2:*], dimension(2:5) :: a, b(4,4), c[10,*], d(4,4)[10,*]
end
!Expect: m4.mod
!module m4
! real(4)::a(2_8:5_8)[2_8:*]
! real(4)::b(1_8:4_8,1_8:4_8)[2_8:*]
! real(4)::c(2_8:5_8)[1_8:10_8,1_8:*]
! real(4)::d(1_8:4_8,1_8:4_8)[1_8:10_8,1_8:*]
!end
