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

// CHECK-HLX-ON: "-target-feature" "+hlx"
// CHECK-HLX-OFF: "-target-feature" "-hlx"
// CHECK-HLX-DEFAULT-NOT: "-target-feature" "{{[+-]}}hlx"
