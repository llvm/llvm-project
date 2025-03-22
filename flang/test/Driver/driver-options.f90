! Check support of -m32 and -m64.
! RUN: %flang -target x86_64-pc-win32 -m32 -### - %s 2>&1 | FileCheck -check-prefix=M32 %s
! RUN: %flang -target i386-pc-win32 -m64 -### - %s 2>&1 | FileCheck -check-prefix=M64 %s

! M32: "-triple" "i{{[0-9]}}86-{{.*}}"
! M64: "-triple" "{{[^-]+}}64-{{.*}}"
