! RUN: %python %S/test_errors.py %s %flang_fc1
! Only the intrinsic modules may define named constants whose values are folded
! when their module file is read.
module m
  intrinsic :: __builtin_numeric_storage_size, __builtin_selected_int_kind, &
    __builtin_selected_unsigned_kind, __builtin_selected_real_kind
  !ERROR: Value of named constant 'nss' (__builtin_numeric_storage_size()) cannot be computed as a constant value
  integer, parameter :: nss = __builtin_numeric_storage_size()
  !ERROR: Value of named constant 'ik' (__builtin_selected_int_kind(9_4)) cannot be computed as a constant value
  integer, parameter :: ik = __builtin_selected_int_kind(9)
  !ERROR: Value of named constant 'uk' (__builtin_selected_unsigned_kind(9_4)) cannot be computed as a constant value
  integer, parameter :: uk = __builtin_selected_unsigned_kind(9)
  !ERROR: Value of named constant 'rk' (__builtin_selected_real_kind(18_4,4931_4)) cannot be computed as a constant value
  integer, parameter :: rk = __builtin_selected_real_kind(18, 4931)
end
