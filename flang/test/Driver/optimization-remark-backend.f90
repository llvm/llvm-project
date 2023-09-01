! This file tests backend passes emitted by the -Rpass family of flags
! loop-delete isn't enabled at O0 so we use at least O1

! DEFINE: %{output} = -S -o /dev/null 2>&1

! Check full -Rpass-missed message is emitted
! RUN: %flang %s -O1 -Rpass-missed %{output} 2>&1 | FileCheck %s --check-prefix=MISSED

! Check full -Rpass-analysis message is emitted
! RUN: %flang %s -O1 -Rpass-analysis %{output} 2>&1 | FileCheck %s --check-prefix=ANALYSIS

! MISSED:   remark: {{[0-9]+}} virtual registers copies {{.*}} total copies cost generated in function
! ANALYSIS: remark: BasicBlock:

program forttest
    implicit none
    integer :: n

    do n = 1,2
        print *, ""
    end do

end program forttest
