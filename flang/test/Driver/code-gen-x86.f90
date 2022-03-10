! Test -emit-obj (X86)

! REQUIRES: x86-registered-target, aarch64-registered-target
! UNSUPPORTED: darwin, macos

!-------------
! RUN COMMANDS
!-------------
! RUN: rm -f %t.o
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-obj %s -o %t.o
! RUN: llvm-objdump --triple x86_64-unknown-linux-gnu --disassemble-all %t.o | FileCheck %s --check-prefix=CORRECT_TRIPLE
! RUN: rm -f %t.o
! RUN: %flang --target=x86_64-unknown-linux-gnu -c %s -o %t.o
! RUN: llvm-objdump --triple x86_64-unknown-linux-gnu --disassemble-all %t.o | FileCheck %s --check-prefix=CORRECT_TRIPLE

! RUN: %flang -c --target=x86_64-unknown-linux-gnu %s -o %t.o
! RUN: llvm-objdump --triple aarch64-unknown-linux-gnu --disassemble-all %t.o | FileCheck %s --check-prefix=INCORRECT_TRIPLE

!----------------
! EXPECTED OUTPUT
!----------------
! CORRECT_TRIPLE-LABEL: <_QQmain>:
! CORRECT_TRIPLE-NEXT:  	retq

! When incorrect triple is used to disassemble, there won't be a ret(q) instruction at all.
! INCORRECT_TRIPLE-LABEL: <_QQmain>:
! INCORRECT_TRIPLE-NOT:  	ret

!------
! INPUT
!------
end program
