// Check that -fno-store-merging is forwarded to -cc1 and that the default
// (or explicit -fstore-merging) does not pass the flag.

// RUN: %clang -### -c %s -fstore-merging -fno-store-merging 2>&1 | FileCheck %s
// CHECK: "-fno-store-merging"

// RUN: %clang -### -c %s 2>&1 | FileCheck --check-prefix=DEFAULT %s
// RUN: %clang -### -c %s -fno-store-merging -fstore-merging 2>&1 | FileCheck --check-prefix=DEFAULT %s
// DEFAULT-NOT: "-fno-store-merging"
