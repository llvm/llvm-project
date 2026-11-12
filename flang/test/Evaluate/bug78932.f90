!RUN: not %flang_fc1 %s 2>&1 | FileCheck %s
!CHECK: error: Actual argument for 'a1=' has bad type 'CHARACTER(KIND=1,LEN=1_8)'
real, parameter :: bad_amax0 = amax0('a', 'b')
end
