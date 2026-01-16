// REQUIRES: aarch64-registered-target

// RUN: %clang -### -c --target=aarch64 %s 2>&1 | FileCheck %s --check-prefix NONE
// NONE:     "-cc1"
// NONE-NOT: "-fptrauth-

//// -fptauth-* driver flags on Linux are only supported with pauthtest ABI.
// RUN: %clang -### -c --target=aarch64-linux -mabi=pauthtest \
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
// RUN:   -fno-ptrauth-elf-got -fptrauth-elf-got \
// RUN:   -fno-aarch64-jump-table-hardening -faarch64-jump-table-hardening \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=ALL-LINUX-PAUTHABI
// RUN: %clang -### -c --target=aarch64-linux-pauthtest \
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
// RUN:   -fno-ptrauth-elf-got -fptrauth-elf-got \
// RUN:   -fno-aarch64-jump-table-hardening -faarch64-jump-table-hardening \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=ALL-LINUX-PAUTHABI
// ALL-LINUX-PAUTHABI: "-cc1"{{.*}} "-fptrauth-intrinsics" "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-type-info-vtable-pointer-discrimination" "-fptrauth-indirect-gotos" "-fptrauth-init-fini" "-fptrauth-init-fini-address-discrimination" "-fptrauth-elf-got"{{.*}} "-faarch64-jump-table-hardening"

// RUN: %clang -### -c --target=aarch64-linux \
// RUN:   -fno-aarch64-jump-table-hardening -faarch64-jump-table-hardening \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=ALL-LINUX
// ALL-LINUX: "-cc1"{{.*}} "-faarch64-jump-table-hardening"

//// Some -fptrauth-* flags are supported for ARM64 Darwin.
// RUN: %clang -### -c --target=arm64-darwin \
// RUN:   -fno-ptrauth-intrinsics -fptrauth-intrinsics \
// RUN:   -fno-ptrauth-calls -fptrauth-calls \
// RUN:   -fno-ptrauth-returns -fptrauth-returns \
// RUN:   -fno-ptrauth-auth-traps -fptrauth-auth-traps \
// RUN:   -fno-ptrauth-vtable-pointer-address-discrimination -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -fno-ptrauth-vtable-pointer-type-discrimination -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fno-ptrauth-type-info-vtable-pointer-discrimination -fptrauth-type-info-vtable-pointer-discrimination \
// RUN:   -fno-ptrauth-indirect-gotos -fptrauth-indirect-gotos \
// RUN:   -fno-aarch64-jump-table-hardening -faarch64-jump-table-hardening \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=ALL-DARWIN
// ALL-DARWIN: "-cc1"{{.*}} "-fptrauth-intrinsics" "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-type-info-vtable-pointer-discrimination" "-fptrauth-indirect-gotos"{{.*}} "-faarch64-jump-table-hardening"

// RUN: %clang -### -c --target=aarch64-linux -mabi=pauthtest %s 2>&1 | FileCheck %s --check-prefix=PAUTHABI1
// RUN: %clang -### -c --target=aarch64-linux-pauthtest %s 2>&1 | FileCheck %s --check-prefix=PAUTHABI1
// PAUTHABI1:      "-cc1"{{.*}} "-triple" "aarch64-unknown-linux-pauthtest"
// PAUTHABI1-SAME: "-fptrauth-intrinsics" "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-type-info-vtable-pointer-discrimination" "-fptrauth-indirect-gotos" "-fptrauth-init-fini" "-fptrauth-init-fini-address-discrimination" "-faarch64-jump-table-hardening"
// PAUTHABI1-SAME: "-target-abi" "pauthtest"
// PAUTHABI1-NOT: "-fptrauth-function-pointer-type-discrimination"

// RUN: %clang -### -c --target=aarch64-linux -mabi=pauthtest -fno-ptrauth-intrinsics \
// RUN:   -fno-ptrauth-calls -fno-ptrauth-returns -fno-ptrauth-auth-traps \
// RUN:   -fno-ptrauth-vtable-pointer-address-discrimination -fno-ptrauth-vtable-pointer-type-discrimination \
// RUN:   -fno-ptrauth-type-info-vtable-pointer-discrimination -fno-ptrauth-indirect-gotos \
// RUN:   -fno-ptrauth-init-fini -fno-ptrauth-init-fini-address-discrimination \
// RUN:   -fno-aarch64-jump-table-hardening %s 2>&1 | FileCheck %s --check-prefix=PAUTHABI2
// RUN: %clang -### -c --target=aarch64-linux-pauthtest -fno-ptrauth-intrinsics \
// RUN:   -fno-ptrauth-calls -fno-ptrauth-returns -fno-ptrauth-auth-traps \
// RUN:   -fno-ptrauth-vtable-pointer-address-discrimination -fno-ptrauth-vtable-pointer-type-discrimination \
// RUN:   -fno-ptrauth-type-info-vtable-pointer-discrimination -fno-ptrauth-indirect-gotos \
// RUN:   -fno-ptrauth-init-fini -fno-ptrauth-init-fini-address-discrimination \
// RUN:   -fno-aarch64-jump-table-hardening %s 2>&1 | FileCheck %s --check-prefix=PAUTHABI2

//// Non-linux OS: pauthtest ABI has no effect in terms of passing ptrauth cc1 flags.
//// An error about unsupported ABI will be emitted later in pipeline (see ERR3 below)
// RUN: %clang -### -c --target=aarch64 -mabi=pauthtest %s 2>&1 | FileCheck %s --check-prefix=PAUTHABI2

// PAUTHABI2:      "-cc1"
// PAUTHABI2-SAME: "-target-abi" "pauthtest"
// PAUTHABI2-NOT:  "-fptrauth-
// PAUTHABI2-NOT: "-faarch64-jump-table-hardening"

//// Non-linux OS: pauthtest environment does not correspond to pauthtest ABI; aapcs is the default.
// RUN: %clang -### -c --target=aarch64-pauthtest %s 2>&1 | FileCheck %s --check-prefix=PAUTHABI3
// PAUTHABI3:      "-cc1"
// PAUTHABI3-SAME: "-target-abi" "aapcs"
// PAUTHABI3-NOT:  "-fptrauth-
// PAUTHABI3-NOT: "-faarch64-jump-table-hardening"

//// Non-pauthtest ABI.
// RUN: not %clang -### -c --target=aarch64-linux -fptrauth-intrinsics -fptrauth-calls -fptrauth-returns -fptrauth-auth-traps \
// RUN:   -fptrauth-vtable-pointer-address-discrimination -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-type-info-vtable-pointer-discrimination -fptrauth-indirect-gotos -fptrauth-init-fini \
// RUN:   -fptrauth-init-fini-address-discrimination -fptrauth-elf-got %s 2>&1 | FileCheck %s --check-prefix=ERR1
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
// ERR1-NEXT: error: unsupported option '-fptrauth-elf-got' for target '{{.*}}'

//// Non-AArch64.
// RUN: not %clang -### -c --target=x86_64-linux -faarch64-jump-table-hardening %s 2>&1 | FileCheck %s --check-prefix=ERR2
// ERR2: error: unsupported option '-faarch64-jump-table-hardening' for target '{{.*}}'

//// Only support PAuth ABI for Linux as for now.
// RUN: not %clang -c --target=aarch64 -mabi=pauthtest %s 2>&1 | FileCheck %s --check-prefix=ERR3
// ERR3: error: unknown target ABI 'pauthtest'

//// The ABI is not specified explicitly, and for non-Linux pauthtest environment does not correspond
//// to pauthtest ABI (each OS target defines this behavior separately). Do not emit an error.
// RUN: %clang -c --target=aarch64-pauthtest %s -o /dev/null

//// PAuth ABI is encoded as environment part of the triple, so don't allow to explicitly set other environments.
// RUN: not %clang -### -c --target=aarch64-linux-gnu -mabi=pauthtest %s 2>&1 | FileCheck %s --check-prefix=ERR4
// ERR4: error: unsupported option '-mabi=pauthtest' for target 'aarch64-unknown-linux-gnu'
// RUN: %clang -### -c --target=aarch64-linux-pauthtest -mabi=pauthtest %s

//// The only branch protection option compatible with PAuthABI is BTI.
// RUN: not %clang -### -c --target=aarch64-linux -mabi=pauthtest -mbranch-protection=pac-ret %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR5_1
// RUN: not %clang -### -c --target=aarch64-linux-pauthtest       -mbranch-protection=pac-ret %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR5_1
// RUN: not %clang -### -c --target=aarch64 -fptrauth-returns     -mbranch-protection=pac-ret %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR5_2
// ERR5_1: error: unsupported option '-mbranch-protection=pac-ret' for target 'aarch64-unknown-linux-pauthtest'
// ERR5_2: error: the combination of '-mbranch-protection=pac-ret' and '-fptrauth-returns' is incompatible

// RUN: not %clang -### -c --target=aarch64-linux -mabi=pauthtest -mbranch-protection=gcs %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR6_1
// RUN: not %clang -### -c --target=aarch64-linux-pauthtest       -mbranch-protection=gcs %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR6_1
// RUN: not %clang -### -c --target=aarch64 -fptrauth-returns     -mbranch-protection=gcs %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR6_2
// ERR6_1: error: unsupported option '-mbranch-protection=gcs' for target 'aarch64-unknown-linux-pauthtest'
// ERR6_2: error: the combination of '-mbranch-protection=gcs' and '-fptrauth-returns' is incompatible

// RUN: not %clang -### -c --target=aarch64-linux -mabi=pauthtest -mbranch-protection=standard %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR7_1
// RUN: not %clang -### -c --target=aarch64-linux-pauthtest       -mbranch-protection=standard %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR7_1
// RUN: not %clang -### -c --target=aarch64 -fptrauth-returns     -mbranch-protection=standard %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR7_2
// ERR7_1: error: unsupported option '-mbranch-protection=standard' for target 'aarch64-unknown-linux-pauthtest'
// ERR7_2: error: the combination of '-mbranch-protection=standard' and '-fptrauth-returns' is incompatible

// RUN: not %clang -### -c --target=aarch64-linux -mabi=pauthtest -msign-return-address=all %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR8
// RUN: not %clang -### -c --target=aarch64-linux-pauthtest       -msign-return-address=all %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR8
// ERR8: error: unsupported option '-msign-return-address=all' for target 'aarch64-unknown-linux-pauthtest'

// RUN: not %clang -### -c --target=aarch64-linux -mabi=pauthtest -msign-return-address=non-leaf %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR9
// RUN: not %clang -### -c --target=aarch64-linux-pauthtest       -msign-return-address=non-leaf %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR9
// ERR9: error: unsupported option '-msign-return-address=non-leaf' for target 'aarch64-unknown-linux-pauthtest'

// RUN: %clang -### -c --target=aarch64-linux -mabi=pauthtest -msign-return-address=none %s
// RUN: %clang -### -c --target=aarch64-linux-pauthtest       -msign-return-address=none %s
// RUN: %clang -### -c --target=aarch64-linux -mabi=pauthtest -mbranch-protection=bti %s
// RUN: %clang -### -c --target=aarch64-linux-pauthtest       -mbranch-protection=bti %s
// RUN: %clang -### -c --target=aarch64-linux -mabi=pauthtest -mbranch-protection=none %s
// RUN: %clang -### -c --target=aarch64-linux-pauthtest       -mbranch-protection=none %s
