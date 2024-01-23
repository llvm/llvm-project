// RUN: %clang --target=x86_64-unknown-linux-gnu -flto -ffat-lto-objects -### %s -c 2>&1 | FileCheck %s -check-prefix=CHECK-CC
// CHECK-CC: -cc1
// CHECK-CC-SAME: -emit-obj
// CHECK-CC-SAME: -ffat-lto-objects

/// Without -flto -S will just emit normal ASM, so we don't expect -emit-{llvm,obj} or -ffat-lto-objects to be passed to cc1.
// RUN: %clang --target=x86_64-unknown-linux-gnu -ffat-lto-objects -### %s -S 2>&1 | FileCheck %s -check-prefix=CHECK-CC-S
// CHECK-CC-S: -cc1
// CHECK-CC-S: -S
// CHECK-CC-S-NOT: -emit-obj
// CHECK-CC-S-NOT: -emit-llvm
// CHECK-CC-S-NOT: -ffat-lto-objects

/// When fat LTO is enabled with -S, we expect asm output and -ffat-lto-objects to be passed to cc1.
// RUN: %clang --target=x86_64-unknown-linux-gnu -flto -ffat-lto-objects -### %s -S 2>&1 | FileCheck %s -check-prefix=CHECK-CC-S-LTO
// CHECK-CC-S-LTO: -cc1
// CHECK-CC-S-NOT: -emit-llvm
// CHECK-CC-S-LTO-SAME: -ffat-lto-objects

/// When fat LTO is enabled with -S and -emit-llvm, we expect IR output and -ffat-lto-objects to be passed to cc1.
// RUN: %clang --target=x86_64-unknown-linux-gnu -flto -ffat-lto-objects -### %s -S -emit-llvm 2>&1 | FileCheck %s -check-prefix=CHECK-CC-S-EL-LTO
// CHECK-CC-S-EL-LTO: -cc1
// CHECK-CC-S-EL-LTO-SAME: -emit-llvm
// CHECK-CC-S-EL-LTO-SAME: -ffat-lto-objects

/// When fat LTO is enabled wihtout -S we expect native object output and -ffat-lto-object to be passed to cc1.
// RUN: %clang --target=x86_64-unknown-linux-gnu -flto -ffat-lto-objects -### %s -c 2>&1 | FileCheck %s -check-prefix=CHECK-CC-C-LTO
// CHECK-CC-C-LTO: -cc1
// CHECK-CC-C-LTO: -emit-obj
// CHECK-CC-C-LTO: -ffat-lto-objects

/// Make sure we don't have a warning for -ffat-lto-objects being unused
// RUN: %clang --target=x86_64-unknown-linux-gnu -ffat-lto-objects -fdriver-only -Werror -v %s -c 2>&1 | FileCheck %s -check-prefix=CHECK-CC-NOLTO
// CHECK-CC-NOLTO: -cc1
// CHECK-CC-NOLTO-SAME: -emit-obj
// CHECK-CC-NOLTO-NOT: -ffat-lto-objects

/// We need to pass an additional flag (--fat-lto-objects) to lld when linking w/ -flto -ffat-lto-objects
/// But it should not be there when LTO is disabled w/ -fno-lto
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -ffat-lto-objects -### 2>&1 | FileCheck --check-prefix=LTO %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -fno-lto -ffat-lto-objects -### 2>&1 | FileCheck --check-prefix=NOLTO %s
// LTO: "--fat-lto-objects"
// NOLTO-NOT: "--fat-lto-objects"
