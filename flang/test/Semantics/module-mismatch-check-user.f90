
! Reject unknown argument option
! RUN: not %flang -fmodule-mismatch-check=invalid %s 2>&1 | FileCheck --check-prefix=INVALID %s
! RUN: not %flang_fc1 -fmodule-mismatch-check=invalid %s 2>&1 | FileCheck --check-prefix=INVALID %s
! INVALID: error: invalid value 'invalid' in '-fmodule-mismatch-check=invalid'


! Test module dependency checksum mismatch handling for user modules
! DEFINE: %{checksum_correct} = -fsyntax-only -I%S/Inputs -I%S/Inputs/dir1
! DEFINE: %{checksum_mismatch} = -fsyntax-only -I%S/Inputs -I%S/Inputs/dir2

! Correct checksum in Inputs/dir1
! RUN: %flang_fc1 %{checksum_correct} %s

! Invalid checksum in Inputs/dir2
! RUN: not %flang_fc1 %{checksum_mismatch} %s 2>&1 | FileCheck --check-prefix=REJECT %s
! RUN: not %flang_fc1 %{checksum_mismatch} -fmodule-mismatch-check=on %s 2>&1 | FileCheck --check-prefix=REJECT %s
! RUN: not %flang_fc1 %{checksum_mismatch} -fmodule-mismatch-check=non-intrinsic %s 2>&1 | FileCheck --check-prefix=REJECT %s
! REJECT: Cannot use module file for module 'module_mismatch_check_a': File is not the right module file for 'module_mismatch_check_a':

! Convert checksum mismatch to warning
! RUN: %flang %{checksum_mismatch} -fmodule-mismatch-check=warn %s 2>&1 | FileCheck --check-prefix=WARNING %s
! WARNING: warning: Module file for module 'module_mismatch_check_a': File has a different checksum than expected for 'module_mismatch_check_a':

! Silence warning using option
! RUN: %flang %{checksum_mismatch} -fmodule-mismatch-check=warn -Wno-module-file-mismatch %s 2>&1 | FileCheck --allow-empty --check-prefix=NO-WARNING %s
! NO-WARNING-NOT: warning


use module_mismatch_check_b
call s2
end
