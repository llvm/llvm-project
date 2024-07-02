// RUN: %clang -### -c --target=aarch64 %s 2>&1 | FileCheck %s --check-prefix NONE
// NONE: "-cc1"
// NONE-NOT: "-fptrauth-

// RUN: %clang -### -c --target=aarch64 \
// RUN:   -fno-ptrauth-intrinsics -fptrauth-intrinsics \
// RUN:   -fno-ptrauth-calls -fptrauth-calls \
// RUN:   -fno-ptrauth-returns -fptrauth-returns \
// RUN:   -fno-ptrauth-auth-traps -fptrauth-auth-traps \
// RUN:   -fno-ptrauth-vtable-pointer-address-discrimination -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -fno-ptrauth-vtable-pointer-type-discrimination -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fno-ptrauth-init-fini -fptrauth-init-fini \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=ALL
// ALL: "-cc1"{{.*}} "-fptrauth-intrinsics" "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-init-fini"

// RUN: %clang -### -c --target=aarch64 -mbranch-protection=pauthabi %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=PAUTHABI1
// PAUTHABI1: "-cc1"{{.*}} "-fptrauth-intrinsics" "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-init-fini"

// RUN: %clang -### -c --target=aarch64 -mbranch-protection=pauthabi -fno-ptrauth-intrinsics \
// RUN:   -fno-ptrauth-calls -fno-ptrauth-returns -fno-ptrauth-auth-traps \
// RUN:   -fno-ptrauth-vtable-pointer-address-discrimination -fno-ptrauth-vtable-pointer-type-discrimination \
// RUN:   -fno-ptrauth-init-fini %s 2>&1 | FileCheck %s --check-prefix=PAUTHABI2
// PAUTHABI2-NOT: "-fptrauth-intrinsics"
// PAUTHABI2-NOT: "-fptrauth-calls"
// PAUTHABI2-NOT: "-fptrauth-returns"
// PAUTHABI2-NOT: "-fptrauth-auth-traps"
// PAUTHABI2-NOT: "-fptrauth-vtable-pointer-address-discrimination"
// PAUTHABI2-NOT: "-fptrauth-vtable-pointer-type-discrimination"
// PAUTHABI2-NOT: "-fptrauth-init-fini"

// RUN: not %clang -### -c --target=x86_64 -fptrauth-intrinsics -fptrauth-calls -fptrauth-returns -fptrauth-auth-traps \
// RUN:   -fptrauth-vtable-pointer-address-discrimination -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-init-fini %s 2>&1 | FileCheck %s --check-prefix=ERR1
// ERR1:      error: unsupported option '-fptrauth-intrinsics' for target '{{.*}}'
// ERR1-NEXT: error: unsupported option '-fptrauth-calls' for target '{{.*}}'
// ERR1-NEXT: error: unsupported option '-fptrauth-returns' for target '{{.*}}'
// ERR1-NEXT: error: unsupported option '-fptrauth-auth-traps' for target '{{.*}}'
// ERR1-NEXT: error: unsupported option '-fptrauth-vtable-pointer-address-discrimination' for target '{{.*}}'
// ERR1-NEXT: error: unsupported option '-fptrauth-vtable-pointer-type-discrimination' for target '{{.*}}'
// ERR1-NEXT: error: unsupported option '-fptrauth-init-fini' for target '{{.*}}'

// RUN: not %clang -### -c --target=x86_64 -mbranch-protection=pauthabi %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR2
// ERR2: error: unsupported option '-mbranch-protection=' for target 'x86_64'
