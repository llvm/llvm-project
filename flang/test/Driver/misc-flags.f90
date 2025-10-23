! Make sure that `-l` is "visible" to Flang's driver
! RUN: %flang -lpgmath -### %s

! Make sure that `-Wl` is "visible" to Flang's driver
! RUN: %flang -Wl,abs -### %s

! Make sure that `-fuse-ld' is "visible" to Flang's driver
! RUN: %flang -fuse-ld= -### %s

! Make sure that `-L' is "visible" to Flang's driver
! RUN: %flang -L/ -### %s

! Check that '-pie' is "visible" to Flang's driver and is passed on to the
! linker.
! RUN: %flang -pie -### %s 2>&1 | FileCheck %s --check-prefix=PIE
! PIE: "-pie"

! Check that '-no-pie' is "visible" to Flang's driver and that "-pie" is *not*
! passed to the linker.
! RUN: %flang -no-pie -### %s 2>&1 | FileCheck %s --check-prefix=NO-PIE
! NO-PIE-NOT: "-pie"

program hello
  write(*,*), "Hello world!"
end program hello
