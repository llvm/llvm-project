! This file tests the -Rpass family of flags (-Rpass, -Rpass-missed
! and -Rpass-analysis)
! loop-delete isn't enabled at O0 so we use at least O1

! DEFINE: %{output} = -emit-llvm -o /dev/null 2>&1

! Check fc1 can handle -Rpass
! RUN: %flang_fc1 %s -O1 -Rpass %{output} 2>&1 | FileCheck %s --check-prefix=REMARKS

! Check that we can override -Rpass= with -Rno-pass.
! RUN: %flang_fc1 %s -O1 -Rpass -Rno-pass %{output} 2>&1 | FileCheck %s --allow-empty --check-prefix=NO-REMARKS

! Check -Rno-pass, -Rno-pass-analysis, -Rno-pass-missed nothing emitted
! RUN: %flang %s -O1 -Rno-pass -S %{output} 2>&1 | FileCheck %s --allow-empty --check-prefix=NO-REMARKS
! RUN: %flang %s -O1 -Rno-pass-missed -S %{output} 2>&1 | FileCheck %s --allow-empty --check-prefix=NO-REMARKS
! RUN: %flang %s -O1 -Rno-pass-analysis -S %{output} 2>&1 | FileCheck %s --allow-empty --check-prefix=NO-REMARKS

! Check valid -Rpass regex
! RUN: %flang %s -O1 -Rpass=loop -S %{output} 2>&1 | FileCheck %s --check-prefix=PASS-REGEX-LOOP-ONLY

! Check valid -Rpass-missed regex
! RUN: %flang %s -O1 -Rpass-missed=loop -S %{output} 2>&1 | FileCheck %s --check-prefix=MISSED-REGEX-LOOP-ONLY

! Check valid -Rpass-analysis regex
! RUN: %flang %s -O1 -Rpass-analysis=loop -S %{output} 2>&1 | FileCheck %s --check-prefix=ANALYSIS-REGEX-LOOP-ONLY

! Check full -Rpass message is emitted
! RUN: %flang %s -O1 -Rpass -S %{output} 2>&1 | FileCheck %s --check-prefix=PASS

! Check full -Rpass-missed message is emitted
! RUN: %flang %s -O1 -Rpass-missed -S %{output} 2>&1 | FileCheck %s --check-prefix=MISSED

! Check full -Rpass-analysis message is emitted
! RUN: %flang %s -O1 -Rpass-analysis -S -o /dev/null 2>&1 | FileCheck %s --check-prefix=ANALYSIS

! REMARKS: remark:
! NO-REMARKS-NOT: remark:


! With plain -Rpass, -Rpass-missed or -Rpass-analysis, we expect remarks related to 2 opportunities (loop vectorisation / loop delete and load hoisting).
! Once we start filtering, this is reduced to 1 one of the loop passes.

! PASS-REGEX-LOOP-ONLY-NOT:     remark: hoisting load
! PASS-REGEX-LOOP-ONLY:         remark: Loop deleted because it is invariant

! MISSED-REGEX-LOOP-ONLY-NOT:   remark: failed to hoist load with loop-invariant address because load is conditionally executed
! MISSED-REGEX-LOOP-ONLY:       remark: loop not vectorized


! ANALYSIS-REGEX-LOOP-ONLY:     remark: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop
! ANALYSIS-REGEX-LOOP-ONLY:     Unknown data dependence.
! ANALYSIS-REGEX-LOOP-ONLY-NOT: remark:{{.*}}: IR instruction count changed from {{[0-9]+}} to {{[0-9]+}}; Delta: {{-?[0-9]+}}

! PASS:                         remark: hoisting load
! PASS:                         remark: Loop deleted because it is invariant

! MISSED:                       remark: failed to hoist load with loop-invariant address because load is conditionally executed
! MISSED:                       remark: loop not vectorized
! MISSED-NOT:                   remark: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop
! MISSED-NOT:                   Unknown data dependence.

! ANALYSIS:                     remark: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop
! ANALYSIS:                     Unknown data dependence.
! ANALYSIS:                     remark: {{.*}}: IR instruction count changed from {{[0-9]+}} to {{[0-9]+}}; Delta: {{-?[0-9]+}}
! ANALYSIS-NOT:                 remark: failed to hoist load with loop-invariant address because load is conditionally executed

subroutine swap_real(a1, a2)
   implicit none

   real, dimension(1:2) :: aR1
   integer :: i, n
   real, intent(inout) :: a1(:), a2(:)
   real :: a

!  Swap
   do i = 1, min(size(a1), size(a2))
      a = a1(i)
      a1(i) = a2(i)
      a2(i) = a
   end do

! Do a random loop to generate a successful loop-delete pass
    do n = 1,2
        aR1(n) = n * 1.34
    end do

end subroutine swap_real
