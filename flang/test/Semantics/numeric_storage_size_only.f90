! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -fdefault-integer-8 -module-dir %t/only-input %t/only-input/test.f90 2>&1 | FileCheck --allow-empty --implicit-check-not=NUMERIC_STORAGE_SIZE %s
! RUN: %flang_fc1 -fsyntax-only -fdefault-real-8 -module-dir %t/only-input %t/only-input/test.f90 2>&1 | FileCheck --allow-empty --implicit-check-not=NUMERIC_STORAGE_SIZE %s
! RUN: %flang_fc1 -fsyntax-only -fdefault-integer-8 -module-dir %t/unused %t/unused/test.f90 2>&1 | FileCheck --allow-empty --implicit-check-not=NUMERIC_STORAGE_SIZE %s
! RUN: %flang_fc1 -fsyntax-only -fdefault-integer-8 -module-dir %t/direct %t/direct/test.f90 2>&1 | FileCheck --check-prefix=TWO --implicit-check-not=NUMERIC_STORAGE_SIZE %s
! RUN: %flang_fc1 -fsyntax-only -fdefault-integer-8 -module-dir %t/repeated %t/repeated/test.f90 2>&1 | FileCheck --check-prefix=TWO --implicit-check-not=NUMERIC_STORAGE_SIZE %s
! RUN: %flang_fc1 -fsyntax-only -fdefault-integer-8 -module-dir %t/renamed-twice %t/renamed-twice/test.f90 2>&1 | FileCheck --check-prefix=TWO --implicit-check-not=NUMERIC_STORAGE_SIZE %s
! RUN: %flang_fc1 -fsyntax-only -fdefault-integer-8 -module-dir %t/homonym %t/homonym/test.f90 2>&1 | FileCheck --allow-empty --implicit-check-not=NUMERIC_STORAGE_SIZE %s
! RUN: not %flang_fc1 -fsyntax-only -fdefault-integer-8 -module-dir %t/rejected %t/rejected/test.f90 2>&1 | FileCheck --check-prefix=REJECTED --implicit-check-not=NUMERIC_STORAGE_SIZE %s
! RUN: %flang_fc1 -fsyntax-only -fdefault-integer-8 -module-dir %t/reexport %t/reexport/test.f90 2>&1 | FileCheck --check-prefix=ONE --implicit-check-not=NUMERIC_STORAGE_SIZE %s

! ONE: warning: NUMERIC_STORAGE_SIZE from ISO_FORTRAN_ENV is not well-defined because compiler options make default INTEGER(KIND=8) and REAL(KIND=4) have different storage sizes (8 and 4 bytes, respectively) [-Wfolding-value-checks]
! ONE: USE-associated here
! TWO-COUNT-2: warning: NUMERIC_STORAGE_SIZE from ISO_FORTRAN_ENV is not well-defined because compiler options make default INTEGER(KIND=8) and REAL(KIND=4) have different storage sizes (8 and 4 bytes, respectively) [-Wfolding-value-checks]
! REJECTED: error: Reference to 'numeric_storage_size' is ambiguous

!--- only-input/test.f90
subroutine only_input_unit
  use, intrinsic :: iso_fortran_env, only: renamed_input_unit => input_unit
  implicit none
  integer :: i
  i = renamed_input_unit
end subroutine only_input_unit

!--- unused/test.f90
subroutine unused_numeric_storage_size
  use, intrinsic :: iso_fortran_env
  implicit none
  integer :: i
  i = input_unit
end subroutine unused_numeric_storage_size

!--- direct/test.f90
subroutine only_numeric_storage_size
  use, intrinsic :: iso_fortran_env, only: numeric_storage_size
  implicit none
  integer, parameter :: nss = numeric_storage_size
end subroutine only_numeric_storage_size

subroutine renamed_numeric_storage_size
  use, intrinsic :: iso_fortran_env, only: local_nss => numeric_storage_size
  implicit none
  integer, parameter :: nss = local_nss
end subroutine renamed_numeric_storage_size

!--- repeated/test.f90
subroutine repeated_same_import
  use, intrinsic :: iso_fortran_env, only: numeric_storage_size
  use, intrinsic :: iso_fortran_env, only: numeric_storage_size
  integer, parameter :: nss = numeric_storage_size + numeric_storage_size
end subroutine repeated_same_import

!--- renamed-twice/test.f90
subroutine repeated_renamed_import
  use, intrinsic :: iso_fortran_env, only: first_nss => numeric_storage_size
  use, intrinsic :: iso_fortran_env, only: second_nss => numeric_storage_size
  integer, parameter :: nss = first_nss + second_nss
end subroutine repeated_renamed_import

!--- homonym/test.f90
module iso_fortran_env
  integer, parameter :: numeric_storage_size = 123
end module iso_fortran_env

subroutine non_intrinsic_homonym
  use, non_intrinsic :: iso_fortran_env, only: numeric_storage_size
  implicit none
  integer, parameter :: nss = numeric_storage_size
end subroutine non_intrinsic_homonym

!--- rejected/test.f90
module conflicting_nss
  integer, parameter :: numeric_storage_size = 456
end module conflicting_nss

subroutine rejected_intrinsic_import
  use conflicting_nss, only: numeric_storage_size
  use, intrinsic :: iso_fortran_env, only: numeric_storage_size
  ! The ambiguous name is not successfully associated with the intrinsic
  ! constant, so referencing it diagnoses only the import error.
  integer, parameter :: nss = numeric_storage_size
end subroutine rejected_intrinsic_import

!--- reexport/test.f90
module reexported_nss
  use, intrinsic :: iso_fortran_env, only: numeric_storage_size
end module reexported_nss

subroutine import_from_reexport
  use reexported_nss, only: numeric_storage_size
  integer, parameter :: nss = numeric_storage_size
end subroutine import_from_reexport
