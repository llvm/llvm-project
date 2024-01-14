// UNSUPPORTED: system-windows

// RUN: %clang -### %s --target=i686-pc-hurd-gnu --sysroot=%S/Inputs/basic_hurd_tree \
// RUN:   --stdlib=platform 2>&1 | FileCheck --check-prefix=CHECK %s
// CHECK:      "-cc1"
// CHECK-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-gnu/10/../../../../include/c++/10"
/// Debian specific - the path component after 'include' is i386-gnu even
/// though the installation is i686-gnu.
// CHECK-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-gnu/10/../../../../include/i386-gnu/c++/10"
// CHECK-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-gnu/10/../../../../include/c++/10/backward"
// CHECK-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-SAME: "-internal-externc-isystem"
// CHECK-SAME: {{^}} "[[SYSROOT]]/usr/include/i386-gnu"
// CHECK-SAME: {{^}} "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-SAME: {{^}} "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// CHECK:      "{{.*}}ld" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-SAME: "-dynamic-linker" "/lib/ld.so"
// CHECK-SAME: "{{.*}}/usr/lib/gcc/i686-gnu/10/crtbegin.o"
// CHECK-SAME: "-L
// CHECK-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/i686-gnu/10"
// CHECK-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc/i686-gnu/10/../../../../lib32"
// CHECK-SAME: {{^}} "-L[[SYSROOT]]/lib/i386-gnu"
// CHECK-SAME: {{^}} "-L[[SYSROOT]]/lib/../lib32"
// CHECK-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/i386-gnu"
// CHECK-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-SAME: {{^}} "-L[[SYSROOT]]/lib"
// CHECK-SAME: {{^}} "-L[[SYSROOT]]/usr/lib"

// RUN: %clang -### %s --target=i686-pc-hurd-gnu --sysroot=%S/Inputs/basic_hurd_tree \
// RUN:   --stdlib=platform -static 2>&1 | FileCheck --check-prefix=CHECK-STATIC %s
// CHECK-STATIC:      "-cc1"
// CHECK-STATIC-SAME: "-static-define"
// CHECK-STATIC-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-STATIC-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-gnu/10/../../../../include/c++/10"
/// Debian specific - the path component after 'include' is i386-gnu even
/// though the installation is i686-gnu.
// CHECK-STATIC-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-gnu/10/../../../../include/i386-gnu/c++/10"
// CHECK-STATIC-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-gnu/10/../../../../include/c++/10/backward"
// CHECK-STATIC-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-STATIC-SAME: "-internal-externc-isystem"
// CHECK-STATIC-SAME: {{^}} "[[SYSROOT]]/usr/include/i386-gnu"
// CHECK-STATIC-SAME: {{^}} "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-STATIC-SAME: {{^}} "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// CHECK-STATIC:      "{{.*}}ld" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-STATIC-SAME: "-static"
// CHECK-STATIC-SAME: "{{.*}}/usr/lib/gcc/i686-gnu/10/crtbeginT.o"
// CHECK-STATIC-SAME: "-L
// CHECK-STATIC-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/i686-gnu/10"
// CHECK-STATIC-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc/i686-gnu/10/../../../../lib32"
// CHECK-STATIC-SAME: {{^}} "-L[[SYSROOT]]/lib/i386-gnu"
// CHECK-STATIC-SAME: {{^}} "-L[[SYSROOT]]/lib/../lib32"
// CHECK-STATIC-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/i386-gnu"
// CHECK-STATIC-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-STATIC-SAME: {{^}} "-L[[SYSROOT]]/lib"
// CHECK-STATIC-SAME: {{^}} "-L[[SYSROOT]]/usr/lib"

// RUN: %clang -### %s --target=i686-pc-hurd-gnu --sysroot=%S/Inputs/basic_hurd_tree \
// RUN:   -shared 2>&1 | FileCheck --check-prefix=CHECK-SHARED %s
// CHECK-SHARED:      "{{.*}}ld" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-SHARED-SAME: "{{.*}}/usr/lib/gcc/i686-gnu/10/crtbeginS.o"
// CHECK-SHARED-SAME: "-L
// CHECK-SHARED-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/i686-gnu/10"
// CHECK-SHARED-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc/i686-gnu/10/../../../../lib32"
// CHECK-SHARED-SAME: {{^}} "-L[[SYSROOT]]/lib/i386-gnu"
// CHECK-SHARED-SAME: {{^}} "-L[[SYSROOT]]/lib/../lib32"
// CHECK-SHARED-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/i386-gnu"
// CHECK-SHARED-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-SHARED-SAME: {{^}} "-L[[SYSROOT]]/lib"
// CHECK-SHARED-SAME: {{^}} "-L[[SYSROOT]]/usr/lib"

// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:     --gcc-toolchain=%S/Inputs/basic_cross_hurd_tree/usr \
// RUN:     --target=i686-pc-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-CROSS %s
// CHECK-CROSS: "-cc1" "-triple" "i686-pc-hurd-gnu"
// CHECK-CROSS: "{{.*}}/Inputs/basic_cross_hurd_tree/usr/lib/gcc/i686-gnu/10/../../../../i686-gnu/bin/as" "--32"
// CHECK-CROSS: "{{.*}}/Inputs/basic_cross_hurd_tree/usr/lib/gcc/i686-gnu/10/../../../../i686-gnu/bin/ld" {{.*}} "-m" "elf_i386"
// CHECK-CROSS: "{{.*}}/Inputs/basic_cross_hurd_tree/usr/lib/gcc/i686-gnu/10/crtbegin.o"
// CHECK-CROSS: "-L{{.*}}/Inputs/basic_cross_hurd_tree/usr/lib/gcc/i686-gnu/10/../../../../i686-gnu/lib"
