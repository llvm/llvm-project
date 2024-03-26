// RUN: %clang -### -c --target=aarch64 -fno-ptrauth-intrinsics -fptrauth-intrinsics %s 2>&1 | FileCheck %s --check-prefix=INTRIN
// INTRIN: "-cc1"{{.*}} "-fptrauth-intrinsics"

// RUN: %clang -### -c --target=aarch64 -fno-ptrauth-calls -fptrauth-calls %s 2>&1 | FileCheck %s --check-prefix=CALL
// CALL: "-cc1"{{.*}} "-fptrauth-calls"

// RUN: %clang -### -c --target=aarch64 -fno-ptrauth-returns -fptrauth-returns %s 2>&1 | FileCheck %s --check-prefix=RETURN
// RETURN: "-cc1"{{.*}} "-fptrauth-returns"

// RUN: %clang -### -c --target=aarch64 -fno-ptrauth-auth-traps -fptrauth-auth-traps %s 2>&1 | FileCheck %s --check-prefix=TRAP
// TRAP: "-cc1"{{.*}} "-fptrauth-auth-traps"

// RUN: %clang -### -c --target=aarch64 -fno-ptrauth-vtable-pointer-address-discrimination \
// RUN:   -fptrauth-vtable-pointer-address-discrimination %s 2>&1 | FileCheck %s --check-prefix=VPTRADDR
// VPTRADDR: "-cc1"{{.*}} "-fptrauth-vtable-pointer-address-discrimination"

// RUN: %clang -### -c --target=aarch64 -fno-ptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-vtable-pointer-type-discrimination %s 2>&1 | FileCheck %s --check-prefix=VPTRTYPE
// VPTRTYPE: "-cc1"{{.*}} "-fptrauth-vtable-pointer-type-discrimination"

// RUN: %clang -### -c --target=aarch64 -fno-ptrauth-init-fini -fptrauth-init-fini %s 2>&1 | FileCheck %s --check-prefix=INITFINI
// INITFINI: "-cc1"{{.*}} "-fptrauth-init-fini"

// RUN: not %clang -### -c --target=x86_64 -fptrauth-intrinsics -fptrauth-calls -fptrauth-returns -fptrauth-auth-traps \
// RUN:   -fptrauth-vtable-pointer-address-discrimination -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-init-fini %s 2>&1 | FileCheck %s --check-prefix=ERR
// ERR:      error: unsupported option '-fptrauth-intrinsics' for target '{{.*}}'
// ERR-NEXT: error: unsupported option '-fptrauth-calls' for target '{{.*}}'
// ERR-NEXT: error: unsupported option '-fptrauth-returns' for target '{{.*}}'
// ERR-NEXT: error: unsupported option '-fptrauth-auth-traps' for target '{{.*}}'
// ERR-NEXT: error: unsupported option '-fptrauth-vtable-pointer-address-discrimination' for target '{{.*}}'
// ERR-NEXT: error: unsupported option '-fptrauth-vtable-pointer-type-discrimination' for target '{{.*}}'
// ERR-NEXT: error: unsupported option '-fptrauth-init-fini' for target '{{.*}}'
