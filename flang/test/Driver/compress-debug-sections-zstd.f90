! Test -gz=zstd (compressed debug sections with zstd) option handling.

! REQUIRES: zstd

! RUN: %flang -### -c -target x86_64-unknown-linux-gnu -gz=zstd %s 2>&1 | FileCheck %s --check-prefix=GZ-ZSTD
! GZ-ZSTD: "-fc1"
! GZ-ZSTD-SAME: "--compress-debug-sections=zstd"

program test
end program test
