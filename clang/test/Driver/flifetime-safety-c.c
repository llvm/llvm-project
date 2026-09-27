/// -flifetime-safety-c is the default
// RUN: %clang -### -c %s 2>&1 | FileCheck --check-prefix=ENABLED %s
// ENABLED-NOT: "-fno-lifetime-safety-c"

// RUN: %clang -### -c %s -flifetime-safety-c -fno-lifetime-safety-c 2>&1 | \
// RUN:   FileCheck --check-prefix=DISABLED %s
// DISABLED: "-fno-lifetime-safety-c"
