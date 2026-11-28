! Check that narrow integer arguments/returns of main and the runtime start
! function carry signext on SystemZ, as required by the ABI verification.

! REQUIRES: systemz-registered-target

! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
! Only the -S run reaches the SystemZ ABI verifier, which also checks calls
! and returns with no CHECK line (assertions builds only).
! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -S %s -o /dev/null

! CHECK: declare void @_FortranAProgramStart(i32 signext, ptr, ptr, ptr)
! CHECK: define signext i32 @main(i32 signext
end program
