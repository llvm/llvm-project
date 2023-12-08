// RUN: %clang --target=x86_64-unknown-linux-gnu -flto -ffat-lto-objects -### %s -c 2>&1 | FileCheck %s -check-prefix=CHECK-CC
// CHECK-CC: -cc1
// CHECK-CC-SAME: -funified-lto
// CHECK-CC-SAME: -emit-obj
// CHECK-CC-SAME: -ffat-lto-objects

/// Without -flto -S will just emit normal ASM, so we don't expect -emit-{llvm,obj} or -ffat-lto-objects to be passed to cc1.
// RUN: %clang --target=x86_64-unknown-linux-gnu -ffat-lto-objects -### %s -S 2>&1 | FileCheck %s -check-prefix=CHECK-CC-S
// CHECK-CC-S: -cc1
// CHECK-CC-S: -S
// CHECK-CC-S-NOT: -emit-obj
// CHECK-CC-S-NOT: -emit-llvm
// CHECK-CC-S-NOT: -ffat-lto-objects

/// When LTO is enabled, we expect LLVM IR output and -ffat-lto-objects to be passed to cc1.
// RUN: %clang --target=x86_64-unknown-linux-gnu -flto -ffat-lto-objects -### %s -S 2>&1 | FileCheck %s -check-prefix=CHECK-CC-S-LTO
// RUN: %clang --target=x86_64-unknown-linux-gnu -flto -ffat-lto-objects -### %s -S -emit-llvm 2>&1 | FileCheck %s -check-prefix=CHECK-CC-S-LTO
// CHECK-CC-S-LTO: -cc1
// CHECK-CC-S-LTO-SAME: -funified-lto
// CHECK-CC-S-LTO-SAME: -emit-llvm
// CHECK-CC-S-LTO-SAME: -ffat-lto-objects

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

/// Make sure that incompatible options emit the correct diagnostics, since -ffat-lto-objects requires -funified-lto
// RUN: %clang -cc1 -triple=x86_64-unknown-linux-gnu -flto -ffat-lto-objects -funified-lto -emit-llvm-only %s  2>&1 | FileCheck %s -check-prefix=UNIFIED --allow-empty
// UNIFIED-NOT: error:

// RUN: not %clang -cc1 -triple=x86_64-unknown-linux-gnu -flto -ffat-lto-objects -emit-llvm-only %s  2>&1 | FileCheck %s -check-prefix=MISSING_UNIFIED
// MISSING_UNIFIED: error: invalid argument '-ffat-lto-objects' only allowed with '-funified-lto'

// RUN: not %clang -cc1 -triple=x86_64-unknown-linux-gnu -flto -fno-unified-lto -ffat-lto-objects -emit-llvm-only %s  2>&1 | FileCheck %s -check-prefix=NO-UNIFIED
// NO-UNIFIED: error: the combination of '-ffat-lto-objects' and '-fno-unified-lto' is incompatible
