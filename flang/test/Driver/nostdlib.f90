! Check that the libraries do not appear when using -nostdlib and -nodefaultlibs

! RUN: %flang -### -nostdlib --target=ppc64le-linux-gnu %s 2>&1 | FileCheck %s
! RUN: %flang -### -nostdlib --target=aarch64-apple-darwin %s 2>&1 | FileCheck %s
! RUN: %flang -### -nostdlib --target=sparc-sun-solaris2.11 %s 2>&1 | FileCheck %s
! RUN: %flang -### -nostdlib --target=x86_64-unknown-freebsd %s 2>&1 | FileCheck %s
! RUN: %flang -### -nostdlib --target=x86_64-unknown-netbsd %s 2>&1 | FileCheck %s
! RUN: %flang -### -nostdlib --target=x86_64-unknown-openbsd %s 2>&1 | FileCheck %s
! RUN: %flang -### -nostdlib --target=x86_64-unknown-dragonfly %s 2>&1 | FileCheck %s
! RUN: %flang -### -nostdlib --target=x86_64-unknown-haiku %s 2>&1 | FileCheck %s
! RUN: %flang -### -nostdlib --target=x86_64-windows-gnu %s 2>&1 | FileCheck %s

! RUN: %flang -### -nodefaultlibs --target=ppc64le-linux-gnu %s 2>&1 | FileCheck %s
! RUN: %flang -### -nodefaultlibs --target=aarch64-apple-darwin %s 2>&1 | FileCheck %s
! RUN: %flang -### -nodefaultlibs --target=sparc-sun-solaris2.11 %s 2>&1 | FileCheck %s
! RUN: %flang -### -nodefaultlibs --target=x86_64-unknown-freebsd %s 2>&1 | FileCheck %s
! RUN: %flang -### -nodefaultlibs --target=x86_64-unknown-netbsd %s 2>&1 | FileCheck %s
! RUN: %flang -### -nodefaultlibs --target=x86_64-unknown-openbsd %s 2>&1 | FileCheck %s
! RUN: %flang -### -nodefaultlibs --target=x86_64-unknown-dragonfly %s 2>&1 | FileCheck %s
! RUN: %flang -### -nodefaultlibs --target=x86_64-unknown-haiku %s 2>&1 | FileCheck %s
! RUN: %flang -### -nodefaultlibs --target=x86_64-windows-gnu %s 2>&1 | FileCheck %s

! -lgcc will not be linked on all platforms, so checking for that is redundant
! in certain cases. But it is not clear that it is worth checking for each
! platform individually.

! CHECK-NOT: "-lFortranRuntime"
! CHECK-NOT: "-lFortranDecimal"
! CHECK-NOT: "-lgcc"
