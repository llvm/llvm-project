// The PATH variable is heavily used when trying to find a linker.
// RUN: env -i LC_ALL=C LD_LIBRARY_PATH="$LD_LIBRARY_PATH" CLANG_NO_DEFAULT_CONFIG=1 \
// RUN:   %clang %s -### -o %t.o --target=i386-unknown-linux \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     --rtlib=platform --unwindlib=platform -no-pie \
// RUN:     2>&1 | FileCheck --check-prefix=CHECK-LD-32 %s
//
// RUN: env -i LC_ALL=C PATH="" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" CLANG_NO_DEFAULT_CONFIG=1 \
// RUN:   %clang %s -### -o %t.o --target=i386-unknown-linux \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     --rtlib=platform --unwindlib=platform -no-pie \
// RUN:     2>&1 | FileCheck --check-prefix=CHECK-LD-32 %s
//
// CHECK-LD-32-NOT: warning:
// CHECK-LD-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-32: "{{.*}}/usr/lib/gcc/i386-unknown-linux/10.2.0{{/|\\\\}}crtbegin.o"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/10.2.0"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/10.2.0/../../../../i386-unknown-linux/lib"
// CHECK-LD-32: "-L[[SYSROOT]]/lib"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib"
