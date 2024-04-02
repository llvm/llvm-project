!! Test that --gcc-install-dir option is working correctly.
!! It does not test cross-compiling (--sysroot), so crtbegin.o, libgcc/compiler-rt, libc, libFortranRuntime, etc. are not supposed to be affected.

! RUN: %flang -### -o %t %s -no-integrated-as -fuse-ld=ld --gcc-install-dir=%S/Inputs/basic_cross_linux_tree/usr/lib/gcc/i386-unknown-linux-gnu/10.2.0 --target=i386-unknown-linux-gnu 2>&1 | FileCheck --check-prefix=CHECK-I386 %s
! CHECK-I386: "-fc1" "-triple" "i386-unknown-linux-gnu" 
! CHECK-I386: "[[PREFIX:[^"]+]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/i386-unknown-linux-gnu/10.2.0/../../../../i386-unknown-linux-gnu/bin{{/|\\\\}}as"
! CHECK-I386: "[[PREFIX]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/i386-unknown-linux-gnu/10.2.0/../../../../i386-unknown-linux-gnu/bin{{/|\\\\}}ld" {{.*}} "-m" "elf_i386"
! CHECK-I386-SAME: "-L[[PREFIX]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/i386-unknown-linux-gnu/10.2.0" 
! CHECK-I386-SAME: "-L[[PREFIX]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/i386-unknown-linux-gnu/10.2.0/../../../../i386-unknown-linux-gnu/lib"

! RUN: %flang -### -o %t %s -no-integrated-as -fuse-ld=ld --gcc-install-dir=%S/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/10.2.0 --target=x86_64-unknown-linux-gnu 2>&1 | FileCheck --check-prefix=CHECK-X86-64 %s
! CHECK-X86-64: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-X86-64: "[[PREFIX:[^"]+]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/10.2.0/../../../../x86_64-unknown-linux-gnu/bin{{/|\\\\}}as" "--64"
! CHECK-X86-64: "[[PREFIX]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/10.2.0/../../../../x86_64-unknown-linux-gnu/bin{{/|\\\\}}ld" {{.*}} "-m" "elf_x86_64"
! CHECK-X86-64-SAME: "-L[[PREFIX]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/10.2.0" 
! CHECK-X86-64-SAME: "-L[[PREFIX]]/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/10.2.0/../../../../x86_64-unknown-linux-gnu/lib"
