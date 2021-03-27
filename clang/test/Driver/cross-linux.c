// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:   --gcc-toolchain=%S/Inputs/basic_cross_linux_tree/usr \
// RUN:   --target=i386-unknown-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-I386 %s
// CHECK-I386: "-cc1" "-triple" "i386-unknown-linux-gnu"
// CHECK-I386: "{{.*}}/Inputs/basic_cross_linux_tree/usr/lib/gcc/i386-unknown-linux-gnu/4.6.0/../../../../i386-unknown-linux-gnu/bin{{/|\\\\}}as" "--32"
// CHECK-I386: "{{.*}}/Inputs/basic_cross_linux_tree/usr/lib/gcc/i386-unknown-linux-gnu/4.6.0/../../../../i386-unknown-linux-gnu/bin{{/|\\\\}}ld" {{.*}} "-m" "elf_i386"
//
// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:   --gcc-toolchain=%S/Inputs/basic_cross_linux_tree/usr \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-X86-64 %s
// CHECK-X86-64: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-X86-64: "{{.*}}/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/4.6.0/../../../../x86_64-unknown-linux-gnu/bin{{/|\\\\}}as" "--64"
// CHECK-X86-64: "{{.*}}/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/4.6.0/../../../../x86_64-unknown-linux-gnu/bin{{/|\\\\}}ld" {{.*}} "-m" "elf_x86_64"
//
// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:   --gcc-toolchain=%S/Inputs/basic_cross_linux_tree/usr \
// RUN:   --target=x86_64-unknown-linux-gnux32 \
// RUN:   | FileCheck --check-prefix=CHECK-X32 %s
// CHECK-X32: "-cc1" "-triple" "x86_64-unknown-linux-gnux32"
// CHECK-X32: "{{.*}}/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/4.6.0/../../../../x86_64-unknown-linux-gnu/bin{{/|\\\\}}as" "--x32"
// CHECK-X32: "{{.*}}/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/4.6.0/../../../../x86_64-unknown-linux-gnu/bin{{/|\\\\}}ld" {{.*}} "-m" "elf32_x86_64"
//
// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:   --gcc-toolchain=%S/Inputs/basic_cross_linux_tree/usr \
// RUN:   --target=x86_64-unknown-linux-gnu -m32 \
// RUN:   | FileCheck --check-prefix=CHECK-I386 %s
//
// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:   --gcc-toolchain=%S/Inputs/basic_cross_linux_tree/usr \
// RUN:   --target=i386-unknown-linux-gnu -m64 \
// RUN:   | FileCheck --check-prefix=CHECK-X86-64 %s
//
// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_32bit_linux_tree/usr \
// RUN:   --target=i386-unknown-linux \
// RUN:   --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-MULTI32-I386 %s
// CHECK-MULTI32-I386: "-cc1" "-triple" "i386-unknown-linux"
// CHECK-MULTI32-I386: "{{.*}}/Inputs/multilib_32bit_linux_tree/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../i386-unknown-linux/bin{{/|\\\\}}as" "--32"
// CHECK-MULTI32-I386: "{{.*}}/Inputs/multilib_32bit_linux_tree/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../i386-unknown-linux/bin{{/|\\\\}}ld"
// CHECK-MULTI32-I386: "--sysroot=[[sysroot:.*/Inputs/basic_linux_tree]]"
// CHECK-MULTI32-I386: "-m" "elf_i386"
// CHECK-MULTI32-I386: "crti.o" "[[gcc_install:.*/Inputs/multilib_32bit_linux_tree/usr/lib/gcc/i386-unknown-linux/4.6.0]]{{/|\\\\}}crtbegin.o"
// CHECK-MULTI32-I386: "-L[[gcc_install]]"
// CHECK-MULTI32-I386: "-L[[gcc_install]]/../../../../i386-unknown-linux/lib/../lib32"
// CHECK-MULTI32-I386: "-L[[gcc_install]]/../../../../i386-unknown-linux/lib"
// CHECK-MULTI32-I386: "-L[[sysroot]]/lib"
// CHECK-MULTI32-I386: "-L[[sysroot]]/usr/lib"
//
// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_32bit_linux_tree/usr \
// RUN:   --target=x86_64-unknown-linux \
// RUN:   --rtlib=platform \
// RUN:   --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-MULTI32-X86-64 %s
// CHECK-MULTI32-X86-64: "-cc1" "-triple" "x86_64-unknown-linux"
// CHECK-MULTI32-X86-64: "{{.*}}/Inputs/multilib_32bit_linux_tree/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../i386-unknown-linux/bin{{/|\\\\}}as" "--64"
// CHECK-MULTI32-X86-64: "{{.*}}/Inputs/multilib_32bit_linux_tree/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../i386-unknown-linux/bin{{/|\\\\}}ld"
// CHECK-MULTI32-X86-64: "--sysroot=[[sysroot:.*/Inputs/basic_linux_tree]]"
// CHECK-MULTI32-X86-64: "-m" "elf_x86_64"
// CHECK-MULTI32-X86-64: "crti.o" "[[gcc_install:.*/Inputs/multilib_32bit_linux_tree/usr/lib/gcc/i386-unknown-linux/4.6.0]]/64{{/|\\\\}}crtbegin.o"
// CHECK-MULTI32-X86-64: "-L[[gcc_install]]/64"
// CHECK-MULTI32-X86-64: "-L[[gcc_install]]/../../../../i386-unknown-linux/lib/../lib64"
// CHECK-MULTI32-X86-64: "-L[[gcc_install]]/../../../../i386-unknown-linux/lib"
// CHECK-MULTI32-X86-64: "-L[[sysroot]]/lib"
// CHECK-MULTI32-X86-64: "-L[[sysroot]]/usr/lib"
//
// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_64bit_linux_tree/usr \
// RUN:   --target=i386-unknown-linux \
// RUN:   --rtlib=platform \
// RUN:   --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-MULTI64-I386 %s
// CHECK-MULTI64-I386: "-cc1" "-triple" "i386-unknown-linux"
// CHECK-MULTI64-I386: "{{.*}}/Inputs/multilib_64bit_linux_tree/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/bin{{/|\\\\}}as" "--32"
// CHECK-MULTI64-I386: "{{.*}}/Inputs/multilib_64bit_linux_tree/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/bin{{/|\\\\}}ld"
// CHECK-MULTI64-I386: "--sysroot=[[sysroot:.*/Inputs/basic_linux_tree]]"
// CHECK-MULTI64-I386: "-m" "elf_i386"
// CHECK-MULTI64-I386: "crti.o" "[[gcc_install:.*/Inputs/multilib_64bit_linux_tree/usr/lib/gcc/x86_64-unknown-linux/4.6.0]]/32{{/|\\\\}}crtbegin.o"
// CHECK-MULTI64-I386: "-L[[gcc_install]]/32"
// CHECK-MULTI64-I386: "-L[[gcc_install]]/../../../../x86_64-unknown-linux/lib/../lib32"
// CHECK-MULTI64-I386: "-L[[gcc_install]]/../../../../x86_64-unknown-linux/lib"
// CHECK-MULTI64-I386: "-L[[sysroot]]/lib"
// CHECK-MULTI64-I386: "-L[[sysroot]]/usr/lib"
//
// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_64bit_linux_tree/usr \
// RUN:   --target=x86_64-unknown-linux \
// RUN:   --rtlib=platform \
// RUN:   --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-MULTI64-X86-64 %s
// CHECK-MULTI64-X86-64: "-cc1" "-triple" "x86_64-unknown-linux"
// CHECK-MULTI64-X86-64: "{{.*}}/Inputs/multilib_64bit_linux_tree/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/bin{{/|\\\\}}as" "--64"
// CHECK-MULTI64-X86-64: "{{.*}}/Inputs/multilib_64bit_linux_tree/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/bin{{/|\\\\}}ld"
// CHECK-MULTI64-X86-64: "--sysroot=[[sysroot:.*/Inputs/basic_linux_tree]]"
// CHECK-MULTI64-X86-64: "-m" "elf_x86_64"
// CHECK-MULTI64-X86-64: "crti.o" "[[gcc_install:.*/Inputs/multilib_64bit_linux_tree/usr/lib/gcc/x86_64-unknown-linux/4.6.0]]{{/|\\\\}}crtbegin.o"
// CHECK-MULTI64-X86-64: "-L[[gcc_install]]"
// CHECK-MULTI64-X86-64: "-L[[gcc_install]]/../../../../x86_64-unknown-linux/lib/../lib64"
// CHECK-MULTI64-X86-64: "-L[[gcc_install]]/../../../../x86_64-unknown-linux/lib"
// CHECK-MULTI64-X86-64: "-L[[sysroot]]/lib"
// CHECK-MULTI64-X86-64: "-L[[sysroot]]/usr/lib"
