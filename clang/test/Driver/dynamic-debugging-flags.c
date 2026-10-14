// Only support x86_64 targets initially.
// RUN: not %clang -c -target aarch64-unknown-unknown -g -fdynamic-debugging -### -o /dev/null %s 2>&1 | FileCheck %s -check-prefix=CHECK-TARGET-ERR
// RUN: %clang -c -target x86_64-unknown-unknown -g -fdynamic-debugging -### -o /dev/null %s 2>&1 | FileCheck %s -check-prefix=CHECK-OK
// CHECK-TARGET-ERR: error: unsupported option '-fdynamic-debugging' for target 'aarch64-unknown-unknown'

// Do not support LTO initially.
// RUN: not %clang -c -target x86_64-unknown-unknown -g -fdynamic-debugging -flto -### -o /dev/null %s 2>&1 | FileCheck %s -check-prefix=CHECK-LTO-ERR
// RUN: not %clang -c -target x86_64-unknown-unknown -g -fdynamic-debugging -flto=full -### -o /dev/null %s 2>&1 | FileCheck %s -check-prefix=CHECK-LTO-ERR
// RUN: not %clang -c -target x86_64-unknown-unknown -g -fdynamic-debugging -flto=thin -### -o /dev/null %s 2>&1 | FileCheck %s -check-prefix=CHECK-LTO-ERR
// RUN: %clang -c -target x86_64-unknown-unknown -g -fdynamic-debugging -flto -fno-lto -### -o /dev/null %s 2>&1 | FileCheck %s -check-prefix=CHECK-OK
// CHECK-LTO-ERR: clang: error: '-fdynamic-debugging' incompatible with '-flto'

// Do not support split dwarf.
// RUN: not %clang -c -target x86_64-unknown-unknown -fdynamic-debugging -g -gsplit-dwarf -### -o /dev/null %s 2>&1 | FileCheck %s -check-prefix=CHECK-DWO-ERR
// RUN: %clang -c -target x86_64-unknown-unknown -fdynamic-debugging -g -gsplit-dwarf -gno-split-dwarf -### -o /dev/null %s 2>&1 | FileCheck %s -check-prefix=CHECK-OK
// CHECK-DWO-ERR: clang: error: '-fdynamic-debugging' incompatible with '-gsplit-dwarf'

// Do not support llvm IR input.
// RUN: not %clang -c -target x86_64-unknown-unknown -fdynamic-debugging -gsplit-dwarf -### -o /dev/null -x ir %s 2>&1 | FileCheck %s -check-prefix=CHECK-LL-ERR
// CHECK-LL-ERR: clang: error: '-fdynamic-debugging' incompatible with IR input

// Warning - requires debug info.
// RUN: %clang -c -target x86_64-unknown-unknown -fdynamic-debugging -gsplit-dwarf -### -o /dev/null %s 2>&1 | FileCheck %s -check-prefix=CHECK-DBG-WARN
// CHECK-DBG-WARN: clang: warning: '-fdynamic-debugging' ignored: requires debug info
// CHECK-DBG-WARN-NOT: -fdynamic-debugging

// Do not support sanitizers initially.
// RUN: not %clang -fsanitize=undefined -c -target x86_64-unknown-unknown -fdynamic-debugging -### -o /dev/null -x ir %s 2>&1 | FileCheck %s -check-prefix=CHECK-SAN-ERR
// CHECK-SAN-ERR: clang: error: '-fdynamic-debugging' incompatible with '-fsanitize=undefined'

// CHECK-OK-NOT: error:
// CHECK-OK-NOT: warning:
