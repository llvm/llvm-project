! This file tests invalid usage of the -Rpass family of flags (-Rpass, -Rpass-missed
! and -Rpass-analysis)
! loop-delete isn't enabled at O0 so we use at least O1

! Check error on invalid regex -Rpass message is emitted
! RUN: not %flang %s -O1 -Rpass=[ -c 2>&1 | FileCheck %s --check-prefix=REGEX-INVALID

! Check "unknown remark option" warning
! RUN: %flang %s -O1 -R -c 2>&1 | FileCheck %s --check-prefix=WARN

! Check "unknown remark option" warning with suggestion
! RUN: %flang %s -O1 -Rpas -c 2>&1 | FileCheck %s --check-prefix=WARN-SUGGEST

! REGEX-INVALID: error: in pattern '-Rpass=[': brackets ([ ]) not balanced
! WARN: warning: unknown remark option '-R'
! WARN-SUGGEST: warning: unknown remark option '-Rpas'; did you mean '-Rpass'?

program forttest
end program forttest
