! Make sure that `-l` is "visible" to Flang's driver
! RUN: %flang -lpgmath -### %s

! Make sure that `-Wl` is "visible" to Flang's driver
! RUN: %flang -Wl,abs -### %s

! Make sure that `-fuse-ld' is "visible" to Flang's driver
! RUN: %flang -fuse-ld= -### %s

! Make sure that `-L' is "visible" to Flang's driver
! RUN: %flang -L/ -### %s

! ------------------------------------------------------------------------------
! Check that '-pie' and '-no-pie' are "visible" to Flang's driver. Check that
! the correct option is added to the link line.
!
! Last match "wins"
! RUN: %flang -target x86_64-pc-linux-gnu -pie -no-pie -### %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=NO-PIE
! RUN: %flang -target x86_64-pc-linux-gnu -no-pie -pie -### %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=PIE
! RUN: %flang -target x86_64-pc-linux-gnu -pie -### %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=PIE
! RUN: %flang -target x86_64-pc-linux-gnu -no-pie -### %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=NO-PIE
!
! Ensure that "-pie" is passed to the linker.
! RUN: %flang -target i386-unknown-freebsd -pie -### %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=PIE
! RUN: %flang -target aarch64-pc-linux-gnu -pie -### %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=PIE
!
! On Musl Linux, PIE is enabled by default, but can be disabled.
! RUN: %flang -target x86_64-linux-musl -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=PIE
! RUN: %flang -target i686-linux-musl -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=PIE
! RUN: %flang -target armv6-linux-musleabihf %s -### 2>&1 \
! RUN:   | FileCheck %s --check-prefix=PIE
! RUN: %flang -target armv7-linux-musleabihf %s -### 2>&1 \
! RUN:   | FileCheck %s --check-prefix=PIE
! RUN: %flang --target=x86_64-linux-musl -no-pie -### 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NO-PIE
!
! On OpenBSD, -pie is not passed to the linker, but can be forced.
! RUN: %flang -target amd64-pc-openbsd -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NO-PIE
! RUN: %flang -target i386-pc-openbsd -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NO-PIE
! RUN: %flang -target aarch64-unknown-openbsd -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NO-PIE
! RUN: %flang -target arm-unknown-openbsd -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NO-PIE
! RUN: %flang -target powerpc-unknown-openbsd -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NO-PIE
! RUN: %flang -target sparc64-unknown-openbsd -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NO-PIE
! RUN: %flang -target i386-pc-openbsd -pie -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=PIE
!
! On FreeBSD, -pie is not passed to the linker, but can be forced.
! RUN: %flang -target amd64-pc-freebsd -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NO-PIE
! RUN: %flang -target i386-pc-freebsd -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NO-PIE
! RUN: %flang -target aarch64-unknown-freebsd -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NO-PIE
! RUN: %flang -target arm-unknown-freebsd -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NO-PIE
! RUN: %flang -target powerpc-unknown-freebsd -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NO-PIE
! RUN: %flang -target sparc64-unknown-freebsd -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NO-PIE
! RUN: %flang -target i386-pc-freebsd -pie -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=PIE
!
! On AIX, -pie is never passed to the linker.
! RUN: %flang -target powerpc64-unknown-aix -### %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=NO-PIE
! RUN: %flang -target powerpc64-unknown-aix -pie -### %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=NO-PIE,UNUSED
! RUN: %flang -target powerpc64-unknown-aix -no-pie -### %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=NO-PIE,UNUSED
!
! On MinGW and Windows, -pie may be specified, but it is ignored.
! RUN: %flang -target aarch64-pc-windows-gnu -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefixes=NO-PIE
! RUN: %flang -target x86_64-pc-windows-gnu -pie -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefixes=NO-PIE,UNUSED
! RUN: %flang -target i686-pc-windows-gnu -no-pie -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefixes=NO-PIE,UNUSED
! RUN: %flang -target aarch64-windows-msvc -### %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=NO-PIE
! RUN: %flang -target aarch64-windows-msvc -pie -### %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=NO-PIE,UNUSED
! RUN: %flang -target aarch64-windows-msvc -no-pie -### %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=NO-PIE,UNUSED
!
! PIE: "-pie"
! NO-PIE-NOT: "-pie"
! UNUSED: warning: argument unused during compilation: '{{(-no)?}}-pie'
! ------------------------------------------------------------------------------

program hello
  write(*,*), "Hello world!"
end program hello
