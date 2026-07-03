! Verify that the -isysroot flag is known to the frontend and, on Darwin,
! is passed on to the linker.

! When DEFAULT_SYSROOT is set -isysroot has no effect.
! REQUIRES: !default_sysroot
! RUN: %flang -### --target=aarch64-apple-darwin -isysroot /path/to/sysroot \
! RUN:        %s 2>&1 | FileCheck %s --check-prefix=CHECK-DARWIN
! RUN: %flang -### --target=aarch64-linux-gnu -isysroot /path/to/sysroot \
! RUN:        %s 2>&1 | FileCheck %s --check-prefix=CHECK-LINUX

! CHECK-DARWIN: "{{.*}}ld{{(64)?(\.lld)?(\.exe)?}}" {{.*}}"-syslibroot" "/path/to/sysroot"
! Unused on Linux.
! CHECK-LINUX: warning: argument unused during compilation: '-isysroot /path/to/sysroot'
! CHECK-LINUX-NOT: /path/to/sysroot
