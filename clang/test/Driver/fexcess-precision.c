// RUN: %clang -### -target i386 -fexcess-precision=fast -c %s 2>&1  \
// RUN:   | FileCheck --check-prefix=CHECK-FAST %s
// RUN: %clang -### -target i386 -fexcess-precision=standard -c %s 2>&1  \
// RUN:   | FileCheck --check-prefix=CHECK-STD %s
// RUN: %clang -### -target i386 -fexcess-precision=16 -c %s 2>&1  \
// RUN:   | FileCheck --check-prefix=CHECK-NONE %s
// RUN: %clang -### -target i386 -fexcess-precision=none -c %s 2>&1  \
// RUN:   | FileCheck --check-prefix=CHECK-ERR-NONE %s

// RUN: %clang -### -target x86_64 -fexcess-precision=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FAST %s
// RUN: %clang -### -target x86_64 -fexcess-precision=standard -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-STD %s
// RUN: %clang -### -target x86_64 -fexcess-precision=16 -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NONE %s
// RUN: %clang -### -target x86_64 -fexcess-precision=none -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-ERR-NONE %s

// RUN: %clang -### -target aarch64 -fexcess-precision=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK %s
// RUN: %clang -### -target aarch64 -fexcess-precision=standard -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK %s
// RUN: %clang -### -target aarch64 -fexcess-precision=16 -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ERR-16 %s
// RUN: %clang -### -target aarch64 -fexcess-precision=none -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ERR-NONE %s

// CHECK-FAST: "-ffloat16-excess-precision=fast"
// CHECK-STD: "-ffloat16-excess-precision=standard"
// CHECK-NONE: "-ffloat16-excess-precision=none"
// CHECK-ERR-NONE: unsupported argument 'none' to option '-fexcess-precision='
// CHECK: "-cc1"
// CHECK-NOT: "-ffloat16-excess-precision=fast"
// CHECK-ERR-16: unsupported argument '16' to option '-fexcess-precision='
