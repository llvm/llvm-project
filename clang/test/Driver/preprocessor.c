// RUN: %clang -E -x c-header %s > %t
// RUN: grep 'B B' %t

#define A B
A A

// The driver should pass preprocessor dump flags (-dD, -dM and -dI) to cc1 invocation
// RUN: %clang -### -E -dD %s 2>&1 | FileCheck --check-prefix=CHECK-dD %s
// RUN: %clang -### -E -dM %s 2>&1 | FileCheck --check-prefix=CHECK-dM %s
// RUN: %clang -### -E -dI %s 2>&1 | FileCheck --check-prefix=CHECK-dI %s
// CHECK-dD: clang{{.*}} "-cc1" {{.*}} "-dD"
// CHECK-dM: clang{{.*}} "-cc1" {{.*}} "-dM"
// CHECK-dI: clang{{.*}} "-cc1" {{.*}} "-dI"

