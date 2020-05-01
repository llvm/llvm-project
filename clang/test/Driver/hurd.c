// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:     --target=i386-pc-gnu \
// RUN:     --sysroot=%S/Inputs/basic_hurd_tree \
// RUN:   | FileCheck --check-prefix=CHECK %s
// CHECK-NOT: warning:
// CHECK: "-cc1"
// CHECK: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/i386-gnu"
// CHECK: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// CHECK: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK: "-dynamic-linker" "/lib/ld.so"
// CHECK: "{{.*}}/usr/lib/gcc/i386-gnu/4.6.0{{/|\\\\}}crtbegin.o"
// CHECK: "-L[[SYSROOT]]/lib/i386-gnu"
// CHECK: "-L[[SYSROOT]]/lib/../lib32"
// CHECK: "-L[[SYSROOT]]/usr/lib/i386-gnu"
// CHECK: "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK: "-L[[SYSROOT]]/lib"
// CHECK: "-L[[SYSROOT]]/usr/lib"

// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:     --target=i386-pc-gnu -static \
// RUN:     --sysroot=%S/Inputs/basic_hurd_tree \
// RUN:   | FileCheck --check-prefix=CHECK-STATIC %s
// CHECK-STATIC-NOT: warning:
// CHECK-STATIC: "-cc1"
// CHECK-STATIC: "-static-define"
// CHECK-STATIC: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-STATIC: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-STATIC: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/i386-gnu"
// CHECK-STATIC: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-STATIC: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// CHECK-STATIC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-STATIC: "-static"
// CHECK-STATIC: "{{.*}}/usr/lib/gcc/i386-gnu/4.6.0{{/|\\\\}}crtbeginT.o"
// CHECK-STATIC: "-L[[SYSROOT]]/lib/i386-gnu"
// CHECK-STATIC: "-L[[SYSROOT]]/lib/../lib32"
// CHECK-STATIC: "-L[[SYSROOT]]/usr/lib/i386-gnu"
// CHECK-STATIC: "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-STATIC: "-L[[SYSROOT]]/lib"
// CHECK-STATIC: "-L[[SYSROOT]]/usr/lib"

// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:     --target=i386-pc-gnu -shared \
// RUN:     --sysroot=%S/Inputs/basic_hurd_tree \
// RUN:   | FileCheck --check-prefix=CHECK-SHARED %s
// CHECK-SHARED-NOT: warning:
// CHECK-SHARED: "-cc1"
// CHECK-SHARED: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-SHARED: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-SHARED: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/i386-gnu"
// CHECK-SHARED: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-SHARED: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// CHECK-SHARED: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-SHARED: "{{.*}}/usr/lib/gcc/i386-gnu/4.6.0{{/|\\\\}}crtbeginS.o"
// CHECK-SHARED: "-L[[SYSROOT]]/lib/i386-gnu"
// CHECK-SHARED: "-L[[SYSROOT]]/lib/../lib32"
// CHECK-SHARED: "-L[[SYSROOT]]/usr/lib/i386-gnu"
// CHECK-SHARED: "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-SHARED: "-L[[SYSROOT]]/lib"
// CHECK-SHARED: "-L[[SYSROOT]]/usr/lib"

// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:     --gcc-toolchain=%S/Inputs/basic_cross_hurd_tree/usr \
// RUN:     --target=i386-pc-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-CROSS %s
// CHECK-CROSS-NOT: warning:
// CHECK-CROSS: "-cc1" "-triple" "i386-pc-hurd-gnu"
// CHECK-CROSS: "{{.*}}/Inputs/basic_cross_hurd_tree/usr/lib/gcc/i386-gnu/4.6.0/../../../../i386-gnu/bin{{/|\\\\}}as" "--32"
// CHECK-CROSS: "{{.*}}/Inputs/basic_cross_hurd_tree/usr/lib/gcc/i386-gnu/4.6.0/../../../../i386-gnu/bin{{/|\\\\}}ld" {{.*}} "-m" "elf_i386"
// CHECK-CROSS: "{{.*}}/Inputs/basic_cross_hurd_tree/usr/lib/gcc/i386-gnu/4.6.0{{/|\\\\}}crtbegin.o"
// CHECK-CROSS: "-L{{.*}}/Inputs/basic_cross_hurd_tree/usr/lib/gcc/i386-gnu/4.6.0/../../../../i386-gnu/lib"
