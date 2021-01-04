// RUN: %clang -target powerpc-unknown -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE32 %s
// RUN: %clang -target powerpc-unknown -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE32 %s
// RUN: %clang -target powerpcle-unknown -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE32 %s
// CHECK-BE32: "-cc1"{{.*}} "-triple" "powerpc-{{.*}}"

// RUN: %clang -target powerpcle-unknown -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-LE32 %s
// RUN: %clang -target powerpcle-unknown -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-LE32 %s
// RUN: %clang -target powerpc-unknown -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-LE32 %s
// CHECK-LE32: "-cc1"{{.*}} "-triple" "powerpcle-{{.*}}"

// RUN: %clang -target powerpc64-unknown -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE64 %s
// RUN: %clang -target powerpc64-unknown -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE64 %s
// RUN: %clang -target powerpc64le-unknown -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE64 %s
// CHECK-BE64: "-cc1"{{.*}} "-triple" "powerpc64-{{.*}}"

// RUN: %clang -target powerpc64le-unknown -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-LE64 %s
// RUN: %clang -target powerpc64le-unknown -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-LE64 %s
// RUN: %clang -target powerpc64-unknown -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-LE64 %s
// CHECK-LE64: "-cc1"{{.*}} "-triple" "powerpc64le-{{.*}}"
