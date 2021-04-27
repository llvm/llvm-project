// UNSUPPORTED: system-windows

// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:     --target=i686-pc-hurd-gnu \
// RUN:     --sysroot=%S/Inputs/basic_hurd_tree \
// RUN:   | FileCheck --check-prefix=CHECK %s
// CHECK: "-cc1"
// CHECK: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-gnu/4.6.0/../../../../include/c++/4.6.0"
/// Debian specific - the path component after 'include' is i386-gnu even
/// though the installation is i686-gnu.
// CHECK-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-gnu/4.6.0/../../../../include/i386-gnu/c++/4.6.0"
// CHECK-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-gnu/4.6.0/../../../../include/c++/4.6.0/backward"
// CHECK-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK: "-internal-externc-isystem"
// CHECK-SAME: {{^}} "[[SYSROOT]]/usr/include/i386-gnu"
// CHECK-SAME: {{^}} "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-SAME: {{^}} "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// CHECK: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK: "-dynamic-linker" "/lib/ld.so"
// CHECK: "{{.*}}/usr/lib/gcc/i686-gnu/4.6.0{{/|\\\\}}crtbegin.o"
// CHECK: "-L
// CHECK-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/i686-gnu/4.6.0"
// CHECK-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc/i686-gnu/4.6.0/../../../../lib32"
// CHECK-SAME: {{^}} "-L[[SYSROOT]]/lib/i386-gnu"
// CHECK-SAME: {{^}} "-L[[SYSROOT]]/lib/../lib32"
// CHECK-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/i386-gnu"
// CHECK-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-SAME: {{^}} "-L[[SYSROOT]]/lib"
// CHECK-SAME: {{^}} "-L[[SYSROOT]]/usr/lib"

// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:     --target=i686-pc-hurd-gnu -static \
// RUN:     --sysroot=%S/Inputs/basic_hurd_tree \
// RUN:   | FileCheck --check-prefix=CHECK-STATIC %s
// CHECK-STATIC: "-cc1"
// CHECK-STATIC: "-static-define"
// CHECK-STATIC: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-STATIC-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-gnu/4.6.0/../../../../include/c++/4.6.0"
/// Debian specific - the path component after 'include' is i386-gnu even
/// though the installation is i686-gnu.
// CHECK-STATIC-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-gnu/4.6.0/../../../../include/i386-gnu/c++/4.6.0"
// CHECK-STATIC-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-gnu/4.6.0/../../../../include/c++/4.6.0/backward"
// CHECK-STATIC-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-STATIC: "-internal-externc-isystem"
// CHECK-STATIC-SAME: {{^}} "[[SYSROOT]]/usr/include/i386-gnu"
// CHECK-STATIC-SAME: {{^}} "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-STATIC-SAME: {{^}} "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// CHECK-STATIC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-STATIC: "-static"
// CHECK-STATIC: "{{.*}}/usr/lib/gcc/i686-gnu/4.6.0{{/|\\\\}}crtbeginT.o"
// CHECK-STATIC: "-L
// CHECK-STATIC-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/i686-gnu/4.6.0"
// CHECK-STATIC-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc/i686-gnu/4.6.0/../../../../lib32"
// CHECK-STATIC-SAME: {{^}} "-L[[SYSROOT]]/lib/i386-gnu"
// CHECK-STATIC-SAME: {{^}} "-L[[SYSROOT]]/lib/../lib32"
// CHECK-STATIC-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/i386-gnu"
// CHECK-STATIC-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-STATIC-SAME: {{^}} "-L[[SYSROOT]]/lib"
// CHECK-STATIC-SAME: {{^}} "-L[[SYSROOT]]/usr/lib"

// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:     --target=i686-pc-hurd-gnu -shared \
// RUN:     --sysroot=%S/Inputs/basic_hurd_tree \
// RUN:   | FileCheck --check-prefix=CHECK-SHARED %s
// CHECK-SHARED: "-cc1"
// CHECK-SHARED: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-SHARED-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-gnu/4.6.0/../../../../include/c++/4.6.0"
/// Debian specific - the path component after 'include' is i386-gnu even
/// though the installation is i686-gnu.
// CHECK-SHARED-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-gnu/4.6.0/../../../../include/i386-gnu/c++/4.6.0"
// CHECK-SHARED-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-gnu/4.6.0/../../../../include/c++/4.6.0/backward"
// CHECK-SHARED-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-SHARED: "-internal-externc-isystem"
// CHECK-SHARED-SAME: {{^}} "[[SYSROOT]]/usr/include/i386-gnu"
// CHECK-SHARED-SAME: {{^}} "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-SHARED-SAME: {{^}} "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// CHECK-SHARED: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-SHARED: "{{.*}}/usr/lib/gcc/i686-gnu/4.6.0{{/|\\\\}}crtbeginS.o"
// CHECK-SHARED: "-L
// CHECK-SHARED-SAME: {{^}}[[SYSROOT]]/usr/lib/gcc/i686-gnu/4.6.0"
// CHECK-SHARED-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc/i686-gnu/4.6.0/../../../../lib32"
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
// CHECK-CROSS: "{{.*}}/Inputs/basic_cross_hurd_tree/usr/lib/gcc/i686-gnu/4.6.0/../../../../i686-gnu/bin{{/|\\\\}}as" "--32"
// CHECK-CROSS: "{{.*}}/Inputs/basic_cross_hurd_tree/usr/lib/gcc/i686-gnu/4.6.0/../../../../i686-gnu/bin{{/|\\\\}}ld" {{.*}} "-m" "elf_i386"
// CHECK-CROSS: "{{.*}}/Inputs/basic_cross_hurd_tree/usr/lib/gcc/i686-gnu/4.6.0{{/|\\\\}}crtbegin.o"
// CHECK-CROSS: "-L{{.*}}/Inputs/basic_cross_hurd_tree/usr/lib/gcc/i686-gnu/4.6.0/../../../../i686-gnu/lib"
