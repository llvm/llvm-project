// Check macro definitions
// RUN: %clang -target m68k-unknown-linux -m68000 -dM -E %s | FileCheck --check-prefix=CHECK-MX %s
// CHECK-MX: #define __mc68000 1
// CHECK-MX: #define __mc68000__ 1
// CHECK-MX: #define mc68000 1

// RUN: %clang -target m68k-unknown-linux -m68010 -dM -E %s | FileCheck --check-prefix=CHECK-MX10 %s
// CHECK-MX10: #define __mc68000 1
// CHECK-MX10: #define __mc68000__ 1
// CHECK-MX10: #define __mc68010 1
// CHECK-MX10: #define __mc68010__ 1
// CHECK-MX10: #define mc68000 1
// CHECK-MX10: #define mc68010 1

// RUN: %clang -target m68k-unknown-linux -m68020 -dM -E %s | FileCheck --check-prefix=CHECK-MX20 %s
// CHECK-MX20: #define __mc68000 1
// CHECK-MX20: #define __mc68000__ 1
// CHECK-MX20: #define __mc68020 1
// CHECK-MX20: #define __mc68020__ 1
// CHECK-MX20: #define mc68000 1
// CHECK-MX20: #define mc68020 1

// RUN: %clang -target m68k-unknown-linux -m68030 -dM -E %s | FileCheck --check-prefix=CHECK-MX30 %s
// CHECK-MX30: #define __mc68000 1
// CHECK-MX30: #define __mc68000__ 1
// CHECK-MX30: #define __mc68030 1
// CHECK-MX30: #define __mc68030__ 1
// CHECK-MX30: #define mc68000 1
// CHECK-MX30: #define mc68030 1

// RUN: %clang -target m68k-unknown-linux -m68040 -dM -E %s | FileCheck --check-prefix=CHECK-MX40 %s
// CHECK-MX40: #define __mc68000 1
// CHECK-MX40: #define __mc68000__ 1
// CHECK-MX40: #define __mc68040 1
// CHECK-MX40: #define __mc68040__ 1
// CHECK-MX40: #define mc68000 1
// CHECK-MX40: #define mc68040 1

// RUN: %clang -target m68k-unknown-linux -m68060 -dM -E %s | FileCheck --check-prefix=CHECK-MX60 %s
// CHECK-MX60: #define __mc68000 1
// CHECK-MX60: #define __mc68000__ 1
// CHECK-MX60: #define __mc68060 1
// CHECK-MX60: #define __mc68060__ 1
// CHECK-MX60: #define mc68000 1
// CHECK-MX60: #define mc68060 1
