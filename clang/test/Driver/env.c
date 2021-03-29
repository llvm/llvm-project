// These tests try to ensure that the driver operates reasonably when run with
// a strange environment. Unfortunately, it requires a normal shell and the
// 'env' command that understands arguments, unlike the LIT built-in env.
//
// REQUIRES: shell
// The PATH variable is heavily used when trying to find a linker.
// RUN: env -i LC_ALL=C LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
// RUN:   %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-LD-32 %s
//
// RUN: env -i LC_ALL=C PATH="" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
// RUN:   %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-LD-32 %s
//
// CHECK-LD-32-NOT: warning:
// CHECK-LD-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-32: "{{.*}}/usr/lib/gcc/i386-unknown-linux/4.6.0{{/|\\\\}}crtbegin.o"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../i386-unknown-linux/lib"
// CHECK-LD-32: "-L[[SYSROOT]]/lib"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib"
