!! Test that --gcc-toolchain and --gcc-install-dir options are working as expected.
!! It does not test cross-compiling (--sysroot), so crtbegin.o, libgcc/compiler-rt, libc, libFortranRuntime, etc. are not supposed to be affected.
!! PREFIX is captured twice because the driver escapes backslashes (occuring in Windows paths) in the -### output, but not on the "Selected GCC installation:" line.

! RUN: %flang 2>&1 -### -v -o %t %s -no-integrated-as -fuse-ld=ld --target=i386-unknown-linux-gnu --gcc-install-dir=%S/Inputs/basic_cross_linux_tree/usr/lib/gcc/i386-unknown-linux-gnu/10.2.0 | FileCheck %s --check-prefix=CHECK-I386
! RUN: %flang 2>&1 -### -v -o %t %s -no-integrated-as -fuse-ld=ld --target=i386-unknown-linux-gnu --gcc-toolchain=%S/Inputs/basic_cross_linux_tree/usr                                         | FileCheck %s --check-prefix=CHECK-I386
! CHECK-I386:      Selected GCC installation: [[PREFIX:[^"]+]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/i386-unknown-linux-gnu/10.2.0
! CHECK-I386:      "-fc1" "-triple" "i386-unknown-linux-gnu" 
! CHECK-I386:      "[[PREFIX:[^"]+]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/i386-unknown-linux-gnu/10.2.0/../../../../i386-unknown-linux-gnu/bin{{/|\\\\}}as"
! CHECK-I386:      "[[PREFIX]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/i386-unknown-linux-gnu/10.2.0/../../../../i386-unknown-linux-gnu/bin{{/|\\\\}}ld" {{.*}} "-m" "elf_i386"
! CHECK-I386-SAME: "-L[[PREFIX]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/i386-unknown-linux-gnu/10.2.0" 
! CHECK-I386-SAME: "-L[[PREFIX]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/i386-unknown-linux-gnu/10.2.0/../../../../i386-unknown-linux-gnu/lib"

! RUN: %flang 2>&1 -### -v -o %t %s -no-integrated-as -fuse-ld=ld --target=x86_64-unknown-linux-gnu --gcc-install-dir=%S/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/10.2.0 | FileCheck %s --check-prefix=CHECK-X86-64
! RUN: %flang 2>&1 -### -v -o %t %s -no-integrated-as -fuse-ld=ld --target=x86_64-unknown-linux-gnu --gcc-toolchain=%S/Inputs/basic_cross_linux_tree/usr                                           | FileCheck %s --check-prefix=CHECK-X86-64
! CHECK-X86-64:      Selected GCC installation: [[PREFIX:[^"]+]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/10.2.0
! CHECK-X86-64:      "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-X86-64:      "[[PREFIX:[^"]+]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/10.2.0/../../../../x86_64-unknown-linux-gnu/bin{{/|\\\\}}as" "--64"
! CHECK-X86-64:      "[[PREFIX]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/10.2.0/../../../../x86_64-unknown-linux-gnu/bin{{/|\\\\}}ld" {{.*}} "-m" "elf_x86_64"
! CHECK-X86-64-SAME: "-L[[PREFIX]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/10.2.0" 
! CHECK-X86-64-SAME: "-L[[PREFIX]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/10.2.0/../../../../x86_64-unknown-linux-gnu/lib"
