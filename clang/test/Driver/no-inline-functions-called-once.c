// RUN: %clang -### -c %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// DEFAULT-NOT: "-no-inline-functions-called-once"

// RUN: %clang -### -c -fno-inline-functions-called-once %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// DISABLED: {{.*}} "-mllvm" "-no-inline-functions-called-once"

// RUN: %clang -### -c -fno-inline-functions-called-once -finline-functions-called-once %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REENABLED
// REENABLED-NOT: "-no-inline-functions-called-once"

// RUN: %clang -### -c -finline-functions-called-once -fno-inline-functions-called-once %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DISABLED2
// DISABLED2: {{.*}} "-mllvm" "-no-inline-functions-called-once"
