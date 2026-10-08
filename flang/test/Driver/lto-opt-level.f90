! Test that -O flags are correctly forwarded as -plugin-opt=O<n> or
! -plugin-opt=O<n> (AIX) to the linker when LTO is enabled.

! UNSUPPORTED: system-windows, system-solaris

! RUN: %flang --target=x86_64-unknown-linux-gnu -flto -O -### %s 2>&1 | FileCheck %s --check-prefix=LNX-O1
! RUN: %flang --target=x86_64-unknown-linux-gnu -flto=thin -O -### %s 2>&1 | FileCheck %s --check-prefix=LNX-O1
! RUN: %flang --target=x86_64-unknown-linux-gnu -flto -O1 -### %s 2>&1 | FileCheck %s --check-prefix=LNX-O1
! RUN: %flang --target=x86_64-unknown-linux-gnu -flto -O2 -### %s 2>&1 | FileCheck %s --check-prefix=LNX-O2
! RUN: %flang --target=x86_64-unknown-linux-gnu -flto -O3 -### %s 2>&1 | FileCheck %s --check-prefix=LNX-O3

! RUN: %flang --target=powerpc64-ibm-aix -flto -O  -### %s 2>&1 | FileCheck %s --check-prefix=AIX-O1
! RUN: %flang --target=powerpc64-ibm-aix -flto=thin -O -### %s 2>&1 | FileCheck %s --check-prefix=AIX-O1
! RUN: %flang --target=powerpc64-ibm-aix -flto -O1 -### %s 2>&1 | FileCheck %s --check-prefix=AIX-O1
! RUN: %flang --target=powerpc64-ibm-aix -flto -O2 -### %s 2>&1 | FileCheck %s --check-prefix=AIX-O2
! RUN: %flang --target=powerpc64-ibm-aix -flto -O3 -### %s 2>&1 | FileCheck %s --check-prefix=AIX-O3

! LNX-O1: "-plugin-opt=O1"
! LNX-O2: "-plugin-opt=O2"
! LNX-O3: "-plugin-opt=O3"

! AIX-O1: "-bplugin_opt:-O1"
! AIX-O2: "-bplugin_opt:-O2"
! AIX-O3: "-bplugin_opt:-O3"

program test
end program
