! Check that narrow integer arguments/returns of main and the runtime start
! function carry signext on SystemZ, as required by the ABI verification.

! REQUIRES: systemz-registered-target

! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
! -S run: SystemZ ABI verifier (assertions builds only).
! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -S %s -o /dev/null

! CHECK: declare void @_FortranAProgramStart(i32 signext, ptr, ptr, ptr)
! CHECK: define signext i32 @main(i32 signext
end program
