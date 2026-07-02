! Test to check the option "-fpseudo-probe-for-profiling".

! RUN: %flang -### %s 2>&1 | FileCheck %s --check-prefix=NO-PROBE
! RUN: %flang -### -fpseudo-probe-for-profiling %s 2>&1 | FileCheck %s --check-prefix=PROBE
! RUN: %flang -### -fno-pseudo-probe-for-profiling %s 2>&1 | FileCheck %s --check-prefix=NO-PROBE
! RUN: %flang -### -fpseudo-probe-for-profiling -fno-pseudo-probe-for-profiling %s 2>&1 | FileCheck %s --check-prefix=NO-PROBE
! RUN: %flang -### -fpseudo-probe-for-profiling -fno-pseudo-probe-for-profiling -fpseudo-probe-for-profiling %s 2>&1 | FileCheck %s --check-prefix=PROBE

! PROBE: "-fpseudo-probe-for-profiling"
! NO-PROBE-NOT: "-fpseudo-probe-for-profiling"

subroutine test
    implicit none
    print *, 1
end subroutine test
