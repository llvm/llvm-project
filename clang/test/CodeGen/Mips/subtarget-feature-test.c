// REQUIRES: mips64-registered-target
// RUN: %clang --target=mips64-linux-gnu -mcpu=i6400 -o %t -c %s 2>&1 | FileCheck --allow-empty %s
// CHECK-NOT: {{.*}} is not a recognized feature for this target

// RUN: %clang --target=mips64-linux-gnu -mcpu=i6500 -o %t -c %s 2>&1 | FileCheck --allow-empty %s
// CHECK-NOT: {{.*}} is not a recognized feature for this target
