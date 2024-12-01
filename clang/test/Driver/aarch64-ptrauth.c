// REQUIRES: aarch64-registered-target

// RUN: %clang -### -c --target=aarch64 %s 2>&1 | FileCheck %s --check-prefix NONE
// NONE:     "-cc1"
// NONE-NOT: "-fptrauth-

// RUN: %clang -### -c --target=aarch64 \
// RUN:   -fno-ptrauth-intrinsics -fptrauth-intrinsics \
// RUN:   -fno-ptrauth-calls -fptrauth-calls \
// RUN:   -fno-ptrauth-returns -fptrauth-returns \
// RUN:   -fno-ptrauth-auth-traps -fptrauth-auth-traps \
// RUN:   -fno-ptrauth-vtable-pointer-address-discrimination -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -fno-ptrauth-vtable-pointer-type-discrimination -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fno-ptrauth-type-info-vtable-pointer-discrimination -fptrauth-type-info-vtable-pointer-discrimination \
// RUN:   -fno-ptrauth-indirect-gotos -fptrauth-indirect-gotos \
// RUN:   -fno-ptrauth-init-fini -fptrauth-init-fini \
// RUN:   -fno-ptrauth-init-fini-address-discrimination -fptrauth-init-fini-address-discrimination \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=ALL
// ALL: "-cc1"{{.*}} "-fptrauth-intrinsics" "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-type-info-vtable-pointer-discrimination" "-fptrauth-indirect-gotos" "-fptrauth-init-fini" "-fptrauth-init-fini-address-discrimination"

// RUN: %clang -### -c --target=aarch64-linux -mabi=pauthtest %s 2>&1 | FileCheck %s --check-prefix=PAUTHABI1
// RUN: %clang -### -c --target=aarch64-linux-pauthtest %s 2>&1 | FileCheck %s --check-prefix=PAUTHABI1
// PAUTHABI1:      "-cc1"{{.*}} "-triple" "aarch64-unknown-linux-pauthtest"
// PAUTHABI1-SAME: "-target-abi" "pauthtest"
// PAUTHABI1-SAME: "-fptrauth-intrinsics" "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-indirect-gotos" "-fptrauth-init-fini"

// RUN: %clang -### -c --target=aarch64 -mabi=pauthtest -fno-ptrauth-intrinsics \
// RUN:   -fno-ptrauth-calls -fno-ptrauth-returns -fno-ptrauth-auth-traps \
// RUN:   -fno-ptrauth-vtable-pointer-address-discrimination -fno-ptrauth-vtable-pointer-type-discrimination \
// RUN:   -fno-ptrauth-indirect-gotos -fno-ptrauth-init-fini %s 2>&1 | FileCheck %s --check-prefix=PAUTHABI2
// RUN: %clang -### -c --target=aarch64-pauthtest -fno-ptrauth-intrinsics \
// RUN:   -fno-ptrauth-calls -fno-ptrauth-returns -fno-ptrauth-auth-traps \
// RUN:   -fno-ptrauth-vtable-pointer-address-discrimination -fno-ptrauth-vtable-pointer-type-discrimination \
// RUN:   -fno-ptrauth-indirect-gotos -fno-ptrauth-init-fini %s 2>&1 | FileCheck %s --check-prefix=PAUTHABI2
// PAUTHABI2:     "-cc1"
// PAUTHABI2-NOT: "-fptrauth-

// RUN: not %clang -### -c --target=x86_64 -fptrauth-intrinsics -fptrauth-calls -fptrauth-returns -fptrauth-auth-traps \
// RUN:   -fptrauth-vtable-pointer-address-discrimination -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-type-info-vtable-pointer-discrimination -fptrauth-indirect-gotos -fptrauth-init-fini \
// RUN:   -fptrauth-init-fini-address-discrimination %s 2>&1 | FileCheck %s --check-prefix=ERR1
// ERR1:      error: unsupported option '-fptrauth-intrinsics' for target '{{.*}}'
// ERR1-NEXT: error: unsupported option '-fptrauth-calls' for target '{{.*}}'
// ERR1-NEXT: error: unsupported option '-fptrauth-returns' for target '{{.*}}'
// ERR1-NEXT: error: unsupported option '-fptrauth-auth-traps' for target '{{.*}}'
// ERR1-NEXT: error: unsupported option '-fptrauth-vtable-pointer-address-discrimination' for target '{{.*}}'
// ERR1-NEXT: error: unsupported option '-fptrauth-vtable-pointer-type-discrimination' for target '{{.*}}'
// ERR1-NEXT: error: unsupported option '-fptrauth-type-info-vtable-pointer-discrimination' for target '{{.*}}'
// ERR1-NEXT: error: unsupported option '-fptrauth-indirect-gotos' for target '{{.*}}'
// ERR1-NEXT: error: unsupported option '-fptrauth-init-fini' for target '{{.*}}'
// ERR1-NEXT: error: unsupported option '-fptrauth-init-fini-address-discrimination' for target '{{.*}}'

//// Only support PAuth ABI for Linux as for now.
// RUN: not %clang -o /dev/null -c --target=aarch64-unknown -mabi=pauthtest %s 2>&1 | FileCheck %s --check-prefix=ERR2
// RUN: not %clang -o /dev/null -c --target=aarch64-unknown-pauthtest       %s 2>&1 | FileCheck %s --check-prefix=ERR2
// ERR2: error: ABI 'pauthtest' is not supported for 'aarch64-unknown-unknown-pauthtest'

//// PAuth ABI is encoded as environment part of the triple, so don't allow to explicitly set other environments.
// RUN: not %clang -### -c --target=aarch64-linux-gnu -mabi=pauthtest %s 2>&1 | FileCheck %s --check-prefix=ERR3
// ERR3: error: unsupported option '-mabi=pauthtest' for target 'aarch64-unknown-linux-gnu'
// RUN: %clang -### -c --target=aarch64-linux-pauthtest -mabi=pauthtest %s

//// The only branch protection option compatible with PAuthABI is BTI.
// RUN: not %clang -### -c --target=aarch64-linux -mabi=pauthtest -mbranch-protection=pac-ret %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR4
// RUN: not %clang -### -c --target=aarch64-linux-pauthtest       -mbranch-protection=pac-ret %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR4
// ERR4: error: unsupported option '-mbranch-protection=pac-ret' for target 'aarch64-unknown-linux-pauthtest'

// RUN: not %clang -### -c --target=aarch64-linux -mabi=pauthtest -mbranch-protection=gcs %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR5
// RUN: not %clang -### -c --target=aarch64-linux-pauthtest       -mbranch-protection=gcs %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR5
// ERR5: error: unsupported option '-mbranch-protection=gcs' for target 'aarch64-unknown-linux-pauthtest'

// RUN: not %clang -### -c --target=aarch64-linux -mabi=pauthtest -mbranch-protection=standard %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR6
// RUN: not %clang -### -c --target=aarch64-linux-pauthtest       -mbranch-protection=standard %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR6
// ERR6: error: unsupported option '-mbranch-protection=standard' for target 'aarch64-unknown-linux-pauthtest'

// RUN: not %clang -### -c --target=aarch64-linux -mabi=pauthtest -msign-return-address=all %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR7
// RUN: not %clang -### -c --target=aarch64-linux-pauthtest       -msign-return-address=all %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR7
// ERR7: error: unsupported option '-msign-return-address=all' for target 'aarch64-unknown-linux-pauthtest'

// RUN: not %clang -### -c --target=aarch64-linux -mabi=pauthtest -msign-return-address=non-leaf %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR8
// RUN: not %clang -### -c --target=aarch64-linux-pauthtest       -msign-return-address=non-leaf %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR8
// ERR8: error: unsupported option '-msign-return-address=non-leaf' for target 'aarch64-unknown-linux-pauthtest'

// RUN: %clang -### -c --target=aarch64-linux -mabi=pauthtest -msign-return-address=none %s
// RUN: %clang -### -c --target=aarch64-linux-pauthtest       -msign-return-address=none %s
// RUN: %clang -### -c --target=aarch64-linux -mabi=pauthtest -mbranch-protection=bti %s
// RUN: %clang -### -c --target=aarch64-linux-pauthtest       -mbranch-protection=bti %s
// RUN: %clang -### -c --target=aarch64-linux -mabi=pauthtest -mbranch-protection=none %s
// RUN: %clang -### -c --target=aarch64-linux-pauthtest       -mbranch-protection=none %s
