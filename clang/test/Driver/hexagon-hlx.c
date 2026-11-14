// Test the -mhlx / -mno-hlx driver flags translate to the +hlx / -hlx
// target feature.

// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mhlx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HLX-ON %s
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mno-hlx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HLX-OFF %s

// -mhlx is off by default.
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HLX-DEFAULT %s

// Last flag wins.
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mno-hlx -mhlx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HLX-ON %s
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mhlx -mno-hlx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HLX-OFF %s

// HLX is not supported before v85.
// RUN: not %clang -c %s -### --target=hexagon-unknown-elf -mv79 -mhlx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HLX-V79 %s
// RUN: not %clang -c %s -### --target=hexagon-unknown-elf -mv79 -mno-hlx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-NOHLX-V79 %s
// RUN: not %clang -c %s -### --target=hexagon-unknown-elf -mcpu=hexagon -mhlx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HLX-NOVER %s

// Pass -mhlx to an external assembler.
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mhlx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck -check-prefix=CHECK-AS-HLX %s
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mno-hlx -mhlx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck -check-prefix=CHECK-AS-HLX %s
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mhlx -mno-hlx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck -check-prefix=CHECK-AS-NOHLX %s

// CHECK-HLX-ON: "-target-feature" "+hlx"
// CHECK-HLX-OFF: "-target-feature" "-hlx"
// CHECK-HLX-DEFAULT-NOT: "-target-feature" "{{[+-]}}hlx"
// CHECK-AS-HLX: llvm-mc{{.*}}" "-mhlx"
// CHECK-AS-NOHLX: llvm-mc
// CHECK-AS-NOHLX-NOT: "-mhlx"
// CHECK-HLX-V79: error: unsupported option '-mhlx' for target 'v79'
// CHECK-NOHLX-V79: error: unsupported option '-mno-hlx' for target 'v79'
// CHECK-HLX-NOVER: error: unsupported option '-mhlx' for target ''
