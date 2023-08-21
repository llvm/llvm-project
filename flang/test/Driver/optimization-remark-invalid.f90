! This file tests invalid usage of the -Rpass family of flags (-Rpass, -Rpass-missed
! and -Rpass-analysis)
! loop-delete isn't enabled at O0 so we use at least O1

! Check error on invalid regex -Rpass message is emitted
! RUN: not %flang %s -O1 -Rpass=[ -c 2>&1 | FileCheck %s --check-prefix=REGEX-INVALID


! REGEX-INVALID: error: in pattern '-Rpass=[': brackets ([ ]) not balanced

program forttest
end program forttest
