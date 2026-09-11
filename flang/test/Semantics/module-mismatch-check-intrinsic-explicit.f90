
! Reject unknown argument option
! RUN: not %flang -fmodule-mismatch-check=invalid %s 2>&1 | FileCheck --check-prefix=INVALID %s
! RUN: not %flang_fc1 -fmodule-mismatch-check=invalid %s 2>&1 | FileCheck --check-prefix=INVALID %s
! INVALID: error: invalid value 'invalid' in '-fmodule-mismatch-check=invalid'


! Test module dependency checksum mismatch handling for intrinsic modules
! Inputs/device-side-modules-a.mod records a deliberately wrong checksum for
! its dependency on iso_fortran_env.
! DEFINE: %{checksum_mismatch} = -fsyntax-only -I%S/Inputs

! RUN: not %flang_fc1 %{checksum_mismatch} %s 2>&1 | FileCheck --check-prefix=REJECT %s
! RUN: not %flang_fc1 %{checksum_mismatch} -fmodule-mismatch-check=on %s 2>&1 | FileCheck --check-prefix=REJECT %s
! REJECT: Cannot use module file for module 'iso_fortran_env': File is not the right module file for 'iso_fortran_env'

! RUN: %flang %{checksum_mismatch} -fmodule-mismatch-check=non-intrinsic %s 2>&1 | FileCheck --allow-empty --check-prefix=WARN %s
! RUN: %flang %{checksum_mismatch} -fmodule-mismatch-check=warn %s 2>&1 | FileCheck --check-prefix=WARN %s
! WARN: warning: Module file for module 'iso_fortran_env': File has a different checksum than expected for 'iso_fortran_env':

! RUN: %flang %{checksum_mismatch} -fmodule-mismatch-check=non-intrinsic -Wno-module-file-mismatch %s 2>&1 | FileCheck --allow-empty --check-prefix=NO-WARN %s
! RUN: %flang %{checksum_mismatch} -fmodule-mismatch-check=warn -Wno-module-file-mismatch %s 2>&1 | FileCheck --allow-empty --check-prefix=NO-WARN %s
! NO-WARN-NOT: warning


module module_mismatch_check_intrinsic
 use device_modfile01_a, only: x
end module
