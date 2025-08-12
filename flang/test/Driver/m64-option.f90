! Check support of -m64.
! RUN: %flang -target i386-pc-win32 -m64 -### - %s 2>&1 | FileCheck -check-prefix=M64 %s
! RUN: %flang -target x86_64-linux-gnu -m64 -### - %s 2>&1 | FileCheck -check-prefix=M64 %s
! RUN: %flang -target x86_64-unknown-windows -m64 -### - %s 2>&1 | FileCheck -check-prefix=M64 %s
! RUN: %flang -target x86_64-unknown-macosx -m64 -### - %s 2>&1 | FileCheck -check-prefix=M64 %s

! M64: "-triple" "x86_64-{{.*}}"
