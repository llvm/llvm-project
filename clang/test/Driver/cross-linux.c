// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:   --gcc-toolchain=%S/Inputs/basic_cross_linux_tree/usr \
// RUN:   --target=i386-unknown-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-I386 %s
// CHECK-I386: "-cc1" "-triple" "i386-unknown-linux-gnu"
// CHECK-I386: "{{.*}}/Inputs/basic_cross_linux_tree/usr/lib/gcc/i386-unknown-linux-gnu/10.2.0/../../../../i386-unknown-linux-gnu/bin{{/|\\\\}}as" "--32"
// CHECK-I386: "{{.*}}/Inputs/basic_cross_linux_tree/usr/lib/gcc/i386-unknown-linux-gnu/10.2.0/../../../../i386-unknown-linux-gnu/bin{{/|\\\\}}ld" {{.*}} "-m" "elf_i386"
//
// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:   --gcc-toolchain=%S/Inputs/basic_cross_linux_tree/usr \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-X86-64 %s
// CHECK-X86-64: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-X86-64: "{{.*}}/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/10.2.0/../../../../x86_64-unknown-linux-gnu/bin{{/|\\\\}}as" "--64"
// CHECK-X86-64: "{{.*}}/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/10.2.0/../../../../x86_64-unknown-linux-gnu/bin{{/|\\\\}}ld" {{.*}} "-m" "elf_x86_64"
//
// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:   --gcc-toolchain=%S/Inputs/basic_cross_linux_tree/usr \
// RUN:   --target=x86_64-unknown-linux-gnux32 \
// RUN:   | FileCheck --check-prefix=CHECK-X32 %s
// CHECK-X32: "-cc1" "-triple" "x86_64-unknown-linux-gnux32"
// CHECK-X32: "{{.*}}/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/10.2.0/../../../../x86_64-unknown-linux-gnu/bin{{/|\\\\}}as" "--x32"
// CHECK-X32: "{{.*}}/Inputs/basic_cross_linux_tree/usr/lib/gcc/x86_64-unknown-linux-gnu/10.2.0/../../../../x86_64-unknown-linux-gnu/bin{{/|\\\\}}ld" {{.*}} "-m" "elf32_x86_64"
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
