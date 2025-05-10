! RUN: %flang -### -S -o - -fverbose-asm %s 2>&1 | FileCheck %s --check-prefix=FORWARDING
! FORWARDING: -fverbose-asm

! RUN: %flang -S -o - -fverbose-asm %s | FileCheck %s --check-prefix=VERBOSE
! RUN: %flang_fc1 -S -o - -fverbose-asm %s | FileCheck %s --check-prefix=VERBOSE

! RUN: %flang -S -o - %s | FileCheck %s --check-prefix=QUIET
! RUN: %flang_fc1 -S -o - %s | FileCheck %s --check-prefix=QUIET
! RUN: %flang -S -o - -fverbose-asm -fno-verbose-asm %s | FileCheck %s --check-prefix=QUIET
! RUN: %flang_fc1 -S -o - -fverbose-asm -fno-verbose-asm %s | FileCheck %s --check-prefix=QUIET

! VERBOSE: -- Begin function _QQmain
! QUIET-NOT: -- Begin function _QQmain
program test

end program
