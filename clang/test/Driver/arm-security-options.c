// Check the -mbranch-protection=option

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -c %s -### -mbranch-protection=pac-ret 2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-NON-LEAF --check-prefix=KEY-A --check-prefix=BTE-OFF

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -c %s -### -mbranch-protection=pac-ret+leaf 2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-ALL --check-prefix=KEY-A --check-prefix=BTE-OFF

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -c %s -### -mbranch-protection=pac-ret+leaf+b-key 2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-ALL --check-prefix=KEY-B --check-prefix=BTE-OFF

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -c %s -### -mbranch-protection=pac-ret+b-key 2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-NON-LEAF --check-prefix=KEY-B --check-prefix=BTE-OFF

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -c %s -### -mbranch-protection=bti 2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-OFF --check-prefix=BTE-ON

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -c %s -### -mbranch-protection=bti+pac-ret 2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-NON-LEAF --check-prefix=KEY-A --check-prefix=BTE-ON

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -c %s -### -mbranch-protection=bti+pac-ret+leaf 2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-ALL --check-prefix=KEY-A --check-prefix=BTE-ON

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -c %s -### -mbranch-protection=bti 2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-OFF --check-prefix=BTE-ON

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -c %s -### -mbranch-protection=bti+pac-ret+b-key 2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-NON-LEAF --check-prefix=KEY-B --check-prefix=BTE-ON

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -c %s -### -mbranch-protection=bti+pac-ret+leaf+b-key 2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-ALL --check-prefix=KEY-B --check-prefix=BTE-ON

// -mbranch-protection with standard
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -c %s -### -mbranch-protection=standard 2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-NON-LEAF --check-prefix=KEY-A --check-prefix=BTE-ON

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -c %s -### -mbranch-protection=bar 2>&1 | \
// RUN: FileCheck %s --check-prefix=BAD-BP-PROTECTION

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -c %s -### -mbranch-protection=pac-ret+bti+b-key 2>&1 | \
// RUN: FileCheck %s --check-prefix=BAD-B-KEY-COMBINATION

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -c %s -### -mbranch-protection=pac-ret+bti+leaf 2>&1 | \
// RUN: FileCheck %s --check-prefix=BAD-LEAF-COMBINATION

// -mbranch-protection with supported architectures other than v8.1-m.main
// RUN: %clang -target arm-arm-none-eabi -march=armv8-m.main -c %s -### -mbranch-protection=pac-ret 2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-NON-LEAF --check-prefix=KEY-A --check-prefix=BTE-OFF

// RUN: %clang -target arm-arm-none-eabi -march=armv7-m -c %s -### -mbranch-protection=pac-ret 2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-NON-LEAF --check-prefix=KEY-A --check-prefix=BTE-OFF

// RUN: %clang -target arm-arm-none-eabi -march=armv7e-m -c %s -### -mbranch-protection=pac-ret 2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-NON-LEAF --check-prefix=KEY-A --check-prefix=BTE-OFF

// -mbranch-protection with unsupported architectures
// RUN: %clang -target arm-arm-none-eabi -march=armv6-m -c %s -### -mbranch-protection=bti 2>&1 | \
// RUN: FileCheck %s --check-prefix=INCOMPATIBLE-ARCH

// RUN: %clang -target arm-arm-none-eabi -march=armv8-m.base -c %s -### -mbranch-protection=bti 2>&1 | \
// RUN: FileCheck %s --check-prefix=INCOMPATIBLE-ARCH

// RUN: %clang -target arm-arm-none-eabi -march=armv8-a -c %s -### -mbranch-protection=bti 2>&1 | \
// RUN: FileCheck %s --check-prefix=INCOMPATIBLE-ARCH

// RUN: %clang -target arm-arm-none-eabi -march=armv8-r -c %s -### -mbranch-protection=bti 2>&1 | \
// RUN: FileCheck %s --check-prefix=INCOMPATIBLE-ARCH

// RUN: %clang -target arm-arm-none-eabi -march=armv7-a -c %s -### -mbranch-protection=bti 2>&1 | \
// RUN: FileCheck %s --check-prefix=INCOMPATIBLE-ARCH

/// RUN: %clang -target arm-arm-none-eabi -march=armv7-a -mthumb -c %s -### -mbranch-protection=bti 2>&1 | \
/// RUN: FileCheck %s --check-prefix=INCOMPATIBLE-ARCH

// RUN: %clang -target arm-arm-none-eabi -march=armv7-r -c %s -### -mbranch-protection=bti 2>&1 | \
// RUN: FileCheck %s --check-prefix=INCOMPATIBLE-ARCH

// RA-OFF: "-msign-return-address=none"
// RA-NON-LEAF: "-msign-return-address=non-leaf"
// RA-ALL: "-msign-return-address=all"

// KEY-A: "-msign-return-address-key=a_key"
// KEY-B: "-msign-return-address-key=b_key"

// BTE-OFF-NOT: "-mbranch-target-enforce"
// BTE-ON: "-mbranch-target-enforce"

// BAD-BP-PROTECTION: invalid branch protection option 'bar' in '-mbranch-protection={{.*}}'

// BAD-B-KEY-COMBINATION: invalid branch protection option 'b-key' in '-mbranch-protection={{.*}}'
// BAD-LEAF-COMBINATION: invalid branch protection option 'leaf' in '-mbranch-protection={{.*}}'

// INCOMPATIBLE-ARCH: '-mbranch-protection=' option is incompatible with the '{{.*}}' architecture
