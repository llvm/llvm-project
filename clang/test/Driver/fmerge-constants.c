// RUN: %clang -### -c %s -fno-merge-all-constants -fmerge-all-constants 2>&1 | FileCheck %s
// CHECK: "-fmerge-all-constants"

// RUN: %clang -### -c %s 2>&1 | FileCheck --check-prefix=NO %s
// RUN: %clang -### -c %s -fmerge-all-constants -fno-merge-all-constants 2>&1 | FileCheck --check-prefix=NO %s
// NO-NOT: "-fmerge-all-constants"
