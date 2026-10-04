! RUN: %python %S/test_folding.py %s %flang_fc1 -pedantic
!
! AMAX0/AMIN0 take INTEGER arguments (here explicitly KIND=8, which is an
! f18 extension) and return default REAL. MAX1/MIN1 take REAL arguments
! (here explicitly KIND=8) and return default INTEGER.

module specific_extremums_kind_mismatch
  !WARN: portability: Argument types do not match specific intrinsic 'amax0' requirements; using 'max' generic instead and converting the result to REAL(4) if needed [-Wuse-generic-intrinsic-when-specific-doesnt-match]
  real(4), parameter :: result_amax0 = amax0(1_8, 2_8)
  real(4), parameter :: expected_amax0 = 2.0_4
  logical, parameter :: test_amax0 = result_amax0 .EQ. expected_amax0
  !WARN: portability: Argument types do not match specific intrinsic 'amax0' requirements; using 'max' generic instead and converting the result to REAL(4) if needed [-Wuse-generic-intrinsic-when-specific-doesnt-match]
  logical, parameter :: test_amax0_kind = kind(amax0(1_8, 2_8)) .EQ. kind(expected_amax0)

  !WARN: portability: Argument types do not match specific intrinsic 'amin0' requirements; using 'min' generic instead and converting the result to REAL(4) if needed [-Wuse-generic-intrinsic-when-specific-doesnt-match]
  real(4), parameter :: result_amin0 = amin0(1_8, 2_8)
  real(4), parameter :: expected_amin0 = 1.0_4
  logical, parameter :: test_amin0 = result_amin0 .EQ. expected_amin0
  !WARN: portability: Argument types do not match specific intrinsic 'amin0' requirements; using 'min' generic instead and converting the result to REAL(4) if needed [-Wuse-generic-intrinsic-when-specific-doesnt-match]
  logical, parameter :: test_amin0_kind = kind(amin0(1_8, 2_8)) .EQ. kind(expected_amin0)

  !WARN: portability: Argument types do not match specific intrinsic 'max1' requirements; using 'max' generic instead and converting the result to INTEGER(4) if needed [-Wuse-generic-intrinsic-when-specific-doesnt-match]
  integer(4), parameter :: result_max1 = max1(1.0_8, 2.0_8)
  integer(4), parameter :: expected_max1 = 2_4
  logical, parameter :: test_max1 = result_max1 .EQ. expected_max1
  !WARN: portability: Argument types do not match specific intrinsic 'max1' requirements; using 'max' generic instead and converting the result to INTEGER(4) if needed [-Wuse-generic-intrinsic-when-specific-doesnt-match]
  logical, parameter :: test_max1_kind = kind(max1(1.0_8, 2.0_8)) .EQ. kind(expected_max1)

  !WARN: portability: Argument types do not match specific intrinsic 'min1' requirements; using 'min' generic instead and converting the result to INTEGER(4) if needed [-Wuse-generic-intrinsic-when-specific-doesnt-match]
  integer(4), parameter :: result_min1 = min1(1.0_8, 2.0_8)
  integer(4), parameter :: expected_min1 = 1_4
  logical, parameter :: test_min1 = result_min1 .EQ. expected_min1
  !WARN: portability: Argument types do not match specific intrinsic 'min1' requirements; using 'min' generic instead and converting the result to INTEGER(4) if needed [-Wuse-generic-intrinsic-when-specific-doesnt-match]
  logical, parameter :: test_min1_kind = kind(min1(1.0_8, 2.0_8)) .EQ. kind(expected_min1)
end module
