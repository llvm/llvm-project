// Test UndefinedBehaviorSanitizer ld flags.

// RUN: %clang -fcatch-undefined-behavior %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LINUX %s
// CHECK-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-LINUX-NOT: "-lc"
// CHECK-LINUX: libclang_rt.ubsan-i386.a"
// CHECK-LINUX: "-lpthread"
