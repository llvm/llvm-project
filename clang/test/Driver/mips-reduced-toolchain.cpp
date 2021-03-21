// Check frontend and linker invocations on reduced Debian MIPS toolchain.
// This toolchain icludes O32 ABI only.

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-linux-gnu \
// RUN:     --sysroot=%S/Inputs/debian_reduced_mips_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-MIPS %s
// CHECK-DEBIAN-MIPS: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-MIPS: "{{.*}}/usr/lib/gcc/mips-linux-gnu/4.7{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-MIPS: "-L[[SYSROOT]]/usr/lib/gcc/mips-linux-gnu/4.7"
// CHECK-DEBIAN-MIPS: "-L[[SYSROOT]]/usr/lib/mips-linux-gnu"
// CHECK-DEBIAN-MIPS: "-L[[SYSROOT]]/usr/lib/gcc/mips-linux-gnu/4.7/../../.."
// CHECK-DEBIAN-MIPS: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-MIPS: "-L[[SYSROOT]]/usr/lib"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu \
// RUN:     --sysroot=%S/Inputs/debian_reduced_mips_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-MIPSEL %s
// CHECK-DEBIAN-MIPSEL: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-MIPSEL: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.7{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-MIPSEL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.7"
// CHECK-DEBIAN-MIPSEL: "-L[[SYSROOT]]/usr/lib/mipsel-linux-gnu"
// CHECK-DEBIAN-MIPSEL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.7/../../.."
// CHECK-DEBIAN-MIPSEL: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-MIPSEL: "-L[[SYSROOT]]/usr/lib"
