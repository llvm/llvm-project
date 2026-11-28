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

// Forward HLX features to the external assembler via -mattr=.
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 \
// RUN:  -fno-integrated-as 2>&1 | FileCheck -check-prefix=CHECK-AS-DEFAULT %s
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mhlx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck -check-prefix=CHECK-AS-HLX %s
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mno-hlx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck -check-prefix=CHECK-AS-NOHLX %s

// Normalize the external assembler's feature list: last option wins.
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mno-hlx -mhlx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck -check-prefix=CHECK-AS-LAST-HLX %s
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mhlx -mno-hlx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck -check-prefix=CHECK-AS-LAST-NOHLX %s

// Repeated identical options produce only one feature entry.
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mhlx -mhlx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck -check-prefix=CHECK-AS-LAST-HLX %s
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mno-hlx -mno-hlx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck -check-prefix=CHECK-AS-LAST-NOHLX %s

// CHECK-HLX-ON: "-target-feature" "+hlx"
// CHECK-HLX-OFF: "-target-feature" "-hlx"
// CHECK-HLX-DEFAULT-NOT: "-target-feature" "{{[+-]}}hlx"
// CHECK-AS-DEFAULT: llvm-mc
// CHECK-AS-DEFAULT-NOT: "-mattr={{[^"]*[+-]hlx[^"]*}}"
// CHECK-AS-DEFAULT-NOT: "-mhlx"
// CHECK-AS-HLX: llvm-mc{{.*}}"-mattr={{[^"]*}}+hlx{{[^"]*}}"
// CHECK-AS-HLX-NOT: "-mhlx"
// CHECK-AS-NOHLX: llvm-mc{{.*}}"-mattr={{[^"]*}}-hlx{{[^"]*}}"
// CHECK-AS-NOHLX-NOT: "-mhlx"
// CHECK-AS-LAST-HLX: llvm-mc{{.*}}"-mattr=+hlx,
// CHECK-AS-LAST-HLX-NOT: {{[+-]}}hlx{{[,"]}}
// CHECK-AS-LAST-HLX-NOT: "-mhlx"
// CHECK-AS-LAST-NOHLX: llvm-mc{{.*}}"-mattr=-hlx,
// CHECK-AS-LAST-NOHLX-NOT: {{[+-]}}hlx{{[,"]}}
// CHECK-AS-LAST-NOHLX-NOT: "-mhlx"
// CHECK-HLX-V79: error: unsupported option '-mhlx' for target 'v79'
// CHECK-NOHLX-V79: error: unsupported option '-mno-hlx' for target 'v79'
// CHECK-HLX-NOVER: error: unsupported option '-mhlx' for target ''

// -fhlx-vector-stack-limit= is passed to cc1 and requires -mhlx.
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mhlx \
// RUN:  -fhlx-vector-stack-limit=1024 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK-VSTACK-LIMIT %s
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mno-hlx -mhlx \
// RUN:  -fhlx-vector-stack-limit=1024 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK-VSTACK-LIMIT %s
// RUN: not %clang -c %s -### --target=hexagon-unknown-elf -mv85 \
// RUN:  -fhlx-vector-stack-limit=1024 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK-VSTACK-LIMIT-NOHLX %s
// RUN: not %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mhlx -mno-hlx \
// RUN:  -fhlx-vector-stack-limit=1024 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK-VSTACK-LIMIT-NOHLX %s
// RUN: not %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mhlx \
// RUN:  -fhlx-vector-stack-limit=a 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK-VSTACK-LIMIT-BAD %s
// RUN: not %clang -c %s -### --target=hexagon-unknown-elf -mv85 -mhlx \
// RUN:  -fhlx-vector-stack-limit=-128 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK-VSTACK-LIMIT-NEG %s

// CHECK-VSTACK-LIMIT: "-cc1"{{.*}} "-fhlx-vector-stack-limit=1024"
// CHECK-VSTACK-LIMIT-NOHLX: error: -fhlx-vector-stack-limit= requires HLX, use -mhlx to enable it
// CHECK-VSTACK-LIMIT-BAD: error: invalid value 'a' in '-fhlx-vector-stack-limit=a'
// CHECK-VSTACK-LIMIT-NEG: error: invalid value '-128' in '-fhlx-vector-stack-limit=-128'
