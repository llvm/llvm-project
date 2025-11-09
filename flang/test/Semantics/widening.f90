! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror

! WARNING: Default real literal in REAL(8) context might need a kind suffix, as its rounded value 1.00000001490116119384765625e-1_4 is inexact [-Wreal-constant-widening]
real(8), parameter :: warning1 = 0.1
! WARNING: Default real literal in REAL(8) context might need a kind suffix, as its rounded value 1.10000002384185791015625_4 is inexact [-Wreal-constant-widening]
real(8) :: warning2 = 1.1
real, parameter :: noWarning1 = 2.1
real(8) :: noWarning2 = warning1
real(8) :: noWarning3 = noWarning1
real(8) :: noWarning4 = 3.125 ! exact
real(8) :: noWarning5 = 4.1d0 ! explicit 'd'
real(8) :: noWarning6 = 5.1_4 ! explicit suffix
real(8) :: noWarning7 = real(6.1, 8) ! explicit conversion
real(8) :: noWarning8 = real(7.1d0) ! explicit narrowing conversion
! WARNING: Default real literal in REAL(8) context might need a kind suffix, as its rounded value 8.1000003814697265625_4 is inexact [-Wreal-constant-widening]
real(8) :: warning3 = real(8.1) ! no-op conversion
! WARNING: Default real literal in COMPLEX(8) context might need a kind suffix, as its rounded value (9.1000003814697265625_4,1.01000003814697265625e1_4) is inexact [-Wreal-constant-widening]
complex(8), parameter :: warning4 = (9.1, 10.1)
! WARNING: Default real literal in COMPLEX(8) context might need a kind suffix, as its rounded value (1.11000003814697265625e1_4,1.21000003814697265625e1_4) is inexact [-Wreal-constant-widening]
complex(8) :: warning5 = (11.1, 12.1)
! WARNING: Default real literal in REAL(8) context might need a kind suffix, as its rounded value [REAL(4)::1.31000003814697265625e1_4] is inexact [-Wreal-constant-widening]
real(8) :: warning6(1) = [ 13.1 ]
real(8) warning7
! WARNING: Default real literal in REAL(8) context might need a kind suffix, as its rounded value 1.41000003814697265625e1_4 is inexact [-Wreal-constant-widening]
data warning7/14.1/
type derived
! WARNING: Default real literal in REAL(8) context might need a kind suffix, as its rounded value 1.51000003814697265625e1_4 is inexact [-Wreal-constant-widening]
  real(8) :: warning8 = 15.1
  real(8) :: noWarning9 = real(16.1, 8)
  real :: noWarning10 = 17.1
end type
type(derived) dx
real noWarning11
! WARNING: Default real literal in REAL(8) context might need a kind suffix, as its rounded value 1.81000003814697265625e1_4 is inexact [-Wreal-constant-widening]
warning7 = 18.1
! WARNING: Default real literal in REAL(8) context might need a kind suffix, as its rounded value 1.91000003814697265625e1_4 is inexact [-Wreal-constant-widening]
dx%warning8 = 19.1
dx%noWarning10 = 20.1
! WARNING: Default real literal in REAL(8) context might need a kind suffix, as its rounded value 2.11000003814697265625e1_4 is inexact [-Wreal-constant-widening]
dx = derived(21.1)
dx = derived(22.125)
noWarning11 = 23.1
! WARNING: Default real literal in REAL(8) context might need a kind suffix, as its rounded value 2.41000003814697265625e1_4 is inexact [-Wreal-constant-widening]
print *, [real(8) :: 24.1]
! WARNING: Default real literal in REAL(8) context might need a kind suffix, as its rounded value 2.51000003814697265625e1_4 is inexact [-Wreal-constant-widening]
print *, [real(8) :: noWarning11, 25.1]
print *, [real(8) :: noWarning1] ! ok
end
