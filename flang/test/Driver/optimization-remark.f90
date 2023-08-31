! This file tests the -Rpass family of flags (-Rpass, -Rpass-missed
! and -Rpass-analysis)
! loop-delete isn't enabled at O0 so we use at least O1

! DEFINE: %{output} = -emit-llvm -o /dev/null 2>&1

! Check that we can override -Rpass= with -Rno-pass.
! RUN: %flang_fc1 %s -O1 -Rpass %{output} | FileCheck %s --check-prefix=REMARKS
! RUN: %flang_fc1 %s -O1 -Rpass -Rno-pass  %{output} | FileCheck %s --allow-empty --check-prefix=NO-REMARKS

! Check -Rno-pass, -Rno-pass-analysis, -Rno-pass-missed nothing emitted
! RUN: %flang %s -O1 -Rno-pass -S %{output} | FileCheck %s --allow-empty --check-prefix=NO-REMARKS
! RUN: %flang %s -O1 -Rno-pass-missed -S %{output}  | FileCheck %s --allow-empty --check-prefix=NO-REMARKS
! RUN: %flang %s -O1 -Rno-pass-analysis -S %{output} | FileCheck %s --allow-empty --check-prefix=NO-REMARKS

! Check full -Rpass message is emitted
! RUN: %flang %s -O1 -Rpass  -S %{output} | FileCheck %s

! Check full -Rpass-missed message is emitted
! RUN: %flang %s -O1 -Rpass-missed  -S %{output} | FileCheck %s --check-prefix=REMARKS-MISSED

! Check full -Rpass-analysis message is emitted
! RUN: %flang %s -O1 -Rpass-analysis  -S %{output} | FileCheck %s --check-prefix=REMARKS-ANALYSIS

! CHECK: remark: Loop deleted because it is invariant
! REMARKS-MISSED: {{.*}} will not be inlined into {{.*}} because its definition is unavailable
! REMARKS-MISSED: remark: loop not vectorized
! REMARKS-MISSED-NOT: loop not vectorized: instruction cannot be vectorized
! REMARKS-ANALYSIS: remark: loop not vectorized: instruction cannot be vectorized
! REMARKS-ANALYSIS-NOT: {{.*}} will not be inlined into {{.*}} because its definition is unavailable

! REMARKS: remark:
! NO-REMARKS-NOT: remark:

program forttest
    implicit none
    real, dimension(1:50) :: aR1
    integer :: n

    do n = 1,50
        aR1(n) = n * 1.34
        print *, "hello"
    end do

    do n = 1,50
        aR1(n) = n * 1.34
    end do

end program forttest
