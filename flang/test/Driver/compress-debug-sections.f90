! Test -gz (compressed debug sections) option handling.

! REQUIRES: zlib

! RUN: %flang -### -c -target x86_64-unknown-linux-gnu -g -gz %s 2>&1 | FileCheck %s --check-prefix=GZ
! RUN: %flang -### -c -target x86_64-unknown-linux-gnu -g -gz=zlib %s 2>&1 | FileCheck %s --check-prefix=GZ
! GZ: "-fc1"
! GZ-SAME: "--compress-debug-sections=zlib"

! RUN: %flang -### -c -target x86_64-unknown-linux-gnu -g -gz=none %s 2>&1 | FileCheck %s --check-prefix=GZ-NONE
! GZ-NONE: "-fc1"
! GZ-NONE-SAME: "--compress-debug-sections=none"

! RUN: not %flang -### -c -target x86_64-unknown-linux-gnu -g -gz=invalid %s 2>&1 | FileCheck %s --check-prefix=GZ-INVALID
! GZ-INVALID: error: unsupported argument 'invalid' to option '-gz='

! Test that -gz without -g still passes --compress-debug-sections to fc1.
! RUN: %flang -### -c -target x86_64-unknown-linux-gnu -gz=zlib %s 2>&1 | FileCheck %s --check-prefix=GZ-NO-G
! GZ-NO-G: "-fc1"
! GZ-NO-G-SAME: "--compress-debug-sections=zlib"

program test
end program test
