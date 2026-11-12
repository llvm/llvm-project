! RUN: %flang_fc1 -fsyntax-only -fopenacc-default-none-scalars-strict -Wopenacc-default-none-scalars-strict %s 2>&1 | FileCheck --allow-empty --check-prefix=NO-ACC %s
! RUN: not %flang_fc1 -fsyntax-only -fopenacc %s 2>&1 | FileCheck --check-prefix=LENIENT %s
! RUN: not %flang_fc1 -fsyntax-only -fopenacc -Wopenacc-default-none-scalars-strict %s 2>&1 | FileCheck --check-prefix=LENIENT %s
! RUN: not %flang_fc1 -fsyntax-only -fopenacc -Wno-openacc-default-none-scalars-strict %s 2>&1 | FileCheck --check-prefix=SILENT-LENIENT %s
! RUN: not %flang_fc1 -fsyntax-only -fopenacc -fopenacc-default-none-scalars-strict %s 2>&1 | FileCheck --check-prefix=STRICT %s
! RUN: not %flang_fc1 -fsyntax-only -fopenacc -fopenacc-default-none-scalars-strict -Wopenacc-default-none-scalars-strict %s 2>&1 | FileCheck --check-prefix=STRICT %s
! RUN: not %flang_fc1 -fsyntax-only -fopenacc -fopenacc-default-none-scalars-strict -Wno-openacc-default-none-scalars-strict %s 2>&1 | FileCheck --check-prefix=STRICT %s
! RUN: not %flang_fc1 -fsyntax-only -fopenacc -fno-openacc-default-none-scalars-strict %s 2>&1 | FileCheck --check-prefix=LENIENT %s
! RUN: not %flang_fc1 -fsyntax-only -fopenacc -fno-openacc-default-none-scalars-strict -Wopenacc-default-none-scalars-strict %s 2>&1 | FileCheck --check-prefix=LENIENT %s
! RUN: not %flang_fc1 -fsyntax-only -fopenacc -fno-openacc-default-none-scalars-strict -Wno-openacc-default-none-scalars-strict %s 2>&1 | FileCheck --check-prefix=SILENT-LENIENT %s

! Without -fopenacc, no OpenACC directives are processed, so no diagnostics.
! NO-ACC-NOT: DEFAULT(NONE) clause requires
! NO-ACC-NOT: OpenACC DEFAULT(NONE) ignored for scalar

! With explicit strict mode, all unlisted variables error.
! STRICT: error: The DEFAULT(NONE) clause requires that 's' must be listed in a data-mapping clause
! STRICT-NOT: DEFAULT(NONE) clause requires that 'se'
! STRICT: error: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-mapping clause
! STRICT-NOT: DEFAULT(NONE) clause requires that 'ae'

! By default, and with -fno-openacc-default-none-scalars-strict, implicit scalars
! warn instead of erroring.  Arrays still error.
! LENIENT-NOT: error: The DEFAULT(NONE) clause requires that 's'
! LENIENT: warning: OpenACC DEFAULT(NONE) ignored for scalar 's' (-fno-openacc-default-none-scalars-strict) [-Wopenacc-default-none-scalars-strict]
! LENIENT-NOT: OpenACC DEFAULT(NONE) ignored for scalar 'se'
! LENIENT: error: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-mapping clause
! LENIENT-NOT: DEFAULT(NONE) clause requires that 'ae'

! -Wno-openacc-default-none-scalars-strict suppresses the default scalar warning;
! the array error is unaffected.
! SILENT-LENIENT-NOT: error: The DEFAULT(NONE) clause requires that 's'
! SILENT-LENIENT-NOT: OpenACC DEFAULT(NONE) ignored for scalar 's'
! SILENT-LENIENT: error: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-mapping clause
! SILENT-LENIENT-NOT: DEFAULT(NONE) clause requires that 'ae'

subroutine test()
  integer :: s        ! implicit scalar - no explicit data clause
  integer :: se       ! explicit scalar - has copyin clause
  integer :: a(10)    ! implicit array - no explicit data clause
  integer :: ae(10)   ! explicit array - has copyin clause
  !$acc parallel default(none) copyin(se, ae)
  s = 1
  se = 1
  a(1) = 1
  ae(1) = 1
  !$acc end parallel
end subroutine
