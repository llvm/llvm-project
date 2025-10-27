! Make sure that `-l` is "visible" to Flang's driver
! RUN: %flang -lpgmath -### %s

! Make sure that `-Wl` is "visible" to Flang's driver
! RUN: %flang -Wl,abs -### %s

! Make sure that `-fuse-ld' is "visible" to Flang's driver
! RUN: %flang -fuse-ld= -### %s

! Make sure that `-L' is "visible" to Flang's driver
! RUN: %flang -L/ -### %s

! Check that '-pie' and '-no-pie' are "visible" to Flang's driver. Check that
! the correct option is added to the link line. The default is '-pie'.
! RUN: %flang -### %s 2>&1 | FileCheck %s --check-prefix=PIE
! RUN: %flang -pie -### %s 2>&1 | FileCheck %s --check-prefix=PIE
! RUN: %flang -no-pie -### %s 2>&1 | FileCheck %s --check-prefix=NO-PIE
! RUN: %flang -pie -no-pie -### %s 2>&1 | FileCheck %s --check-prefix=NO-PIE
! RUN: %flang -no-pie -pie -### %s 2>&1 | FileCheck %s --check-prefix=PIE
! PIE: "-pie"
! NO-PIE-NOT: "-pie"

program hello
  write(*,*), "Hello world!"
end program hello
