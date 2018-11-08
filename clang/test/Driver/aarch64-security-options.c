// Check the -msign-return-address= option, which has a required argument to
// select scope.
// RUN: %clang -target aarch64--none-eabi -c %s -### -msign-return-address=none                             2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-OFF      --check-prefix=KEY-A --check-prefix=BTE-OFF

// RUN: %clang -target aarch64--none-eabi -c %s -### -msign-return-address=non-leaf                         2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-NON-LEAF --check-prefix=KEY-A --check-prefix=BTE-OFF

// RUN: %clang -target aarch64--none-eabi -c %s -### -msign-return-address=all                              2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-ALL      --check-prefix=KEY-A --check-prefix=BTE-OFF

// Check that the -msign-return-address= option can also accept the signing key
// to use.

// RUN: %clang -target aarch64--none-eabi -c %s -### -msign-return-address=non-leaf                   2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-NON-LEAF --check-prefix=KEY-B --check-prefix=BTE-OFF

// RUN: %clang -target aarch64--none-eabi -c %s -### -msign-return-address=all                        2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-ALL      --check-prefix=KEY-B --check-prefix=BTE-OFF

// -mbranch-protection with standard
// RUN: %clang -target aarch64--none-eabi -c %s -### -mbranch-protection=standard                                2>&1 | \
// RUN: FileCheck %s --check-prefix=RA-NON-LEAF --check-prefix=KEY-A --check-prefix=BTE-ON

// If the -msign-return-address and -mbranch-protection are both used, the
// right-most one controls return address signing.
// RUN: %clang -target aarch64--none-eabi -c %s -### -msign-return-address=non-leaf -mbranch-protection=none     2>&1 | \
// RUN: FileCheck %s --check-prefix=CONFLICT

// RUN: %clang -target aarch64--none-eabi -c %s -### -mbranch-protection=pac-ret -msign-return-address=none     2>&1 | \
// RUN: FileCheck %s --check-prefix=CONFLICT

// RUN: %clang -target aarch64--none-eabi -c %s -### -msign-return-address=foo     2>&1 | \
// RUN: FileCheck %s --check-prefix=BAD-RA-PROTECTION

// RUN: %clang -target aarch64--none-eabi -c %s -### -mbranch-protection=bar     2>&1 | \
// RUN: FileCheck %s --check-prefix=BAD-BP-PROTECTION

// RA-OFF: "-msign-return-address=none"
// RA-NON-LEAF: "-msign-return-address=non-leaf"
// RA-ALL: "-msign-return-address=all"

// KEY-A: "-msign-return-address-key=a_key"

// BTE-OFF-NOT: "-mbranch-target-enforce"
// BTE-ON: "-mbranch-target-enforce"

// CONFLICT: "-msign-return-address=none"

// BAD-RA-PROTECTION: invalid branch protection option 'foo' in '-msign-return-address={{.*}}'
// BAD-BP-PROTECTION: invalid branch protection option 'bar' in '-mbranch-protection={{.*}}'

// BAD-B-KEY-COMBINATION: invalid branch protection option 'b-key' in '-mbranch-protection={{.*}}'
// BAD-LEAF-COMBINATION: invalid branch protection option 'leaf' in '-mbranch-protection={{.*}}'
