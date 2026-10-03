! Test that -O flags are correctly forwarded as -plugin-opt=O<n> to the linker
! when LTO is enabled.

! UNSUPPORTED: system-windows, system-solaris

! --- Linux / lld ---

! RUN: %flang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree -fuse-ld=lld -flto -O  -### %s 2>&1 | FileCheck %s --check-prefix=LLD-O1
! RUN: %flang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree -fuse-ld=lld -flto=thin -O -### %s 2>&1 | FileCheck %s --check-prefix=LLD-O1

! RUN: %flang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree -fuse-ld=lld -flto -O1 -### %s 2>&1 | FileCheck %s --check-prefix=LLD-O1
! RUN: %flang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree -fuse-ld=lld -flto -O2 -### %s 2>&1 | FileCheck %s --check-prefix=LLD-O2
! RUN: %flang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree -fuse-ld=lld -flto -O3 -### %s 2>&1 | FileCheck %s --check-prefix=LLD-O3

! LLD-O1: "-plugin-opt=O1"
! LLD-O2: "-plugin-opt=O2"
! LLD-O3: "-plugin-opt=O3"

! --- AIX ---
! On AIX the linker uses -bplugin_opt:-O<n> instead of -plugin-opt=O<n>.

! RUN: %flang --target=powerpc64-ibm-aix -flto -O  -### %s 2>&1 | FileCheck %s --check-prefix=AIX-O1
! RUN: %flang --target=powerpc64-ibm-aix -flto=thin -O -### %s 2>&1 | FileCheck %s --check-prefix=AIX-O1

! RUN: %flang --target=powerpc64-ibm-aix -flto -O1 -### %s 2>&1 | FileCheck %s --check-prefix=AIX-O1
! RUN: %flang --target=powerpc64-ibm-aix -flto -O2 -### %s 2>&1 | FileCheck %s --check-prefix=AIX-O2
! RUN: %flang --target=powerpc64-ibm-aix -flto -O3 -### %s 2>&1 | FileCheck %s --check-prefix=AIX-O3

! AIX-O1: "-bplugin_opt:-O1"
! AIX-O2: "-bplugin_opt:-O2"
! AIX-O3: "-bplugin_opt:-O3"

program test
end program
