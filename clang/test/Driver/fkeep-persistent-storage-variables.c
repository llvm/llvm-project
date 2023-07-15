// RUN: %clang -fkeep-persistent-storage-variables -c %s -### 2>&1 | FileCheck %s
// RUN: %clang -fkeep-persistent-storage-variables -fno-keep-persistent-storage-variables -c %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-NOKEEP

// CHECK: "-fkeep-persistent-storage-variables"
// CHECK-NOKEEP-NOT: "-fkeep-persistent-storage-variables"
