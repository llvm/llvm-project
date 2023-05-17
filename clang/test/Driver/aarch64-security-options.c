// Check the -msign-return-address= option, which has a required argument to
// select scope.
// RUN: %clang --target=aarch64--none-eabi -c %s -### -msign-return-address=none                             2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-OFF --check-prefix=KEY --check-prefix=BTE-OFF --check-prefix=WARN

// RUN: %clang --target=aarch64--none-eabi -c %s -### -msign-return-address=non-leaf                         2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-NON-LEAF --check-prefix=KEY-A --check-prefix=BTE-OFF --check-prefix=WARN

// RUN: %clang --target=aarch64--none-eabi -c %s -### -msign-return-address=all                              2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-ALL      --check-prefix=KEY-A --check-prefix=BTE-OFF --check-prefix=WARN

// -mbranch-protection with standard
// RUN: %clang --target=aarch64--none-eabi -c %s -### -mbranch-protection=standard                                2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-NON-LEAF --check-prefix=KEY-A --check-prefix=BTE-ON --check-prefix=WARN

// If the -msign-return-address and -mbranch-protection are both used, the
// right-most one controls return address signing.
// RUN: %clang --target=aarch64--none-eabi -c %s -### -msign-return-address=non-leaf -mbranch-protection=none     2>&1 | \
// RUN: FileCheck %s --check-prefix=CONFLICT --check-prefix=WARN

// RUN: %clang --target=aarch64--none-eabi -c %s -### -mbranch-protection=pac-ret -msign-return-address=none     2>&1 | \
// RUN: FileCheck %s --check-prefix=CONFLICT --check-prefix=WARN

// RUN: %clang --target=aarch64--none-eabi -c %s -### -msign-return-address=foo     2>&1 | \
// RUN: FileCheck %s --check-prefix=BAD-RA-PROTECTION --check-prefix=WARN

// RUN: %clang --target=aarch64--none-eabi -c %s -### -mbranch-protection=bar     2>&1 | \
// RUN: FileCheck %s --check-prefix=BAD-BP-PROTECTION --check-prefix=WARN

// RUN: %clang --target=aarch64--none-eabi -### -o /dev/null -mbranch-protection=standard /dev/null 2>&1 | \
// RUN: FileCheck --allow-empty %s --check-prefix=LINKER-DRIVER

// WARN-NOT: warning: ignoring '-mbranch-protection=' option because the 'aarch64' architecture does not support it [-Wbranch-protection]

// RA-OFF: "-msign-return-address=none"
// RA-NON-LEAF: "-msign-return-address=non-leaf"
// RA-ALL: "-msign-return-address=all"

// KEY-A: "-msign-return-address-key=a_key"
// KEY-NOT: "-msign-return-address-key"

// BTE-OFF-NOT: "-mbranch-target-enforce"
// BTE-ON: "-mbranch-target-enforce"

// CONFLICT: "-msign-return-address=none"

// BAD-RA-PROTECTION: unsupported argument 'foo' to option '-msign-return-address='
// BAD-BP-PROTECTION: unsupported argument 'bar' to option '-mbranch-protection='

// BAD-B-KEY-COMBINATION: unsupported argument 'b-key' to option '-mbranch-protection='
// BAD-LEAF-COMBINATION: unsupported argument 'leaf' to option '-mbranch-protection='

// Check that the linker driver doesn't warn about -mbranch-protection=standard
// as an unused option.
// LINKER-DRIVER-NOT: warning: argument unused during compilation: '-mbranch-protection=standard'
