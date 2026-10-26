! Test -f[no-]plt driver forwarding to flang -fc1.
! The driver forwards -fno-plt only for ELF x86 and AArch64. The default is
! PLT, so -fplt is not forwarded. Other targets warn that the flag is unused.

! RUN: %flang -### --target=x86_64-unknown-linux-gnu %s 2>&1 | FileCheck %s --check-prefixes=FC1,DRIVER-PLT
! RUN: %flang -### --target=x86_64-unknown-linux-gnu -fno-plt %s 2>&1 | FileCheck %s --check-prefixes=FC1,DRIVER-NOPLT
! RUN: %flang -### --target=aarch64-unknown-linux-gnu -fno-plt %s 2>&1 | FileCheck %s --check-prefixes=FC1,DRIVER-NOPLT
! RUN: %flang -### --target=x86_64-unknown-linux-gnu -fplt -fno-plt %s 2>&1 | FileCheck %s --check-prefixes=FC1,DRIVER-NOPLT
! RUN: %flang -### --target=x86_64-unknown-linux-gnu -fplt %s 2>&1 | FileCheck %s --check-prefixes=FC1,DRIVER-PLT
! RUN: %flang -### --target=x86_64-unknown-linux-gnu -fno-plt -fplt %s 2>&1 | FileCheck %s --check-prefixes=FC1,DRIVER-PLT
! RUN: %flang -### --target=powerpc64le-unknown-linux-gnu -fno-plt %s 2>&1 | FileCheck %s --check-prefixes=UNUSED,DRIVER-PLT
! RUN: %flang -### --target=aarch64-unknown-windows-msvc -fno-plt %s 2>&1 | FileCheck %s --check-prefixes=UNUSED,DRIVER-PLT
! RUN: %flang -### --target=x86_64-unknown-windows-msvc -fno-plt %s 2>&1 | FileCheck %s --check-prefixes=UNUSED,DRIVER-PLT

! FC1: "-fc1"
! DRIVER-NOPLT-SAME: "-fno-plt"
! DRIVER-PLT-NOT: "-fno-plt"
! UNUSED: warning: argument unused during compilation: '-fno-plt'
! UNUSED-NOT: "-fno-plt"

end program
