// Check that we can manually enable specific ptrauth features.

// RUN: %clang -target aarch64 -c %s -### 2>&1 | FileCheck %s --check-prefix NONE
// NONE: "-cc1"
// NONE-NOT: "-fptrauth-intrinsics"
// NONE-NOT: "-fptrauth-calls"
// NONE-NOT: "-fptrauth-returns"
// NONE-NOT: "-fptrauth-auth-traps"
// NONE-NOT: "-fptrauth-vtable-pointer-address-discrimination"
// NONE-NOT: "-fptrauth-vtable-pointer-type-discrimination"
// NONE-NOT: "-fptrauth-init-fini"

// RUN: %clang -target aarch64 -fptrauth-intrinsics -c %s -### 2>&1 | FileCheck %s --check-prefix INTRIN
// INTRIN: "-cc1"{{.*}} {{.*}} "-fptrauth-intrinsics"

// RUN: %clang -target aarch64 -fptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix CALL
// CALL: "-cc1"{{.*}} {{.*}} "-fptrauth-calls"

// RUN: %clang -target aarch64 -fptrauth-returns -c %s -### 2>&1 | FileCheck %s --check-prefix RETURN
// RETURN: "-cc1"{{.*}} {{.*}} "-fptrauth-returns"

// RUN: %clang -target aarch64 -fptrauth-auth-traps -c %s -### 2>&1 | FileCheck %s --check-prefix TRAPS
// TRAPS: "-cc1"{{.*}} {{.*}} "-fptrauth-auth-traps"

// RUN: %clang -target aarch64 -fptrauth-vtable-pointer-address-discrimination -c %s -### 2>&1 | FileCheck %s --check-prefix VPTR_ADDR_DISCR
// VPTR_ADDR_DISCR: "-cc1"{{.*}} {{.*}} "-fptrauth-vtable-pointer-address-discrimination"

// RUN: %clang -target aarch64 -fptrauth-vtable-pointer-type-discrimination -c %s -### 2>&1 | FileCheck %s --check-prefix VPTR_TYPE_DISCR
// VPTR_TYPE_DISCR: "-cc1"{{.*}} {{.*}} "-fptrauth-vtable-pointer-type-discrimination"

// RUN: %clang -target aarch64 -fptrauth-init-fini -c %s -### 2>&1 | FileCheck %s --check-prefix INITFINI
// INITFINI: "-cc1"{{.*}} {{.*}} "-fptrauth-init-fini"
