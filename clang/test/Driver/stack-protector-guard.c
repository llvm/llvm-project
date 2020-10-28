// RUN: %clang -### -target x86_64-unknown-unknown -mstack-protector-guard=tls %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-TLS %s
// RUN: %clang -### -target x86_64-unknown-unknown -mstack-protector-guard=global %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-GLOBAL %s
// RUN: %clang -### -target x86_64-unknown-unknown -mstack-protector-guard=local %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-VALUE %s

// CHECK-TLS: "-cc1" {{.*}}"-mstack-protector-guard=tls"
// CHECK-GLOBAL: "-cc1" {{.*}}"-mstack-protector-guard=global"
// INVALID-VALUE: error: invalid value 'local' in 'mstack-protector-guard=','valid arguments to '-mstack-protector-guard=' are:tls global'

// RUN: %clang -### -target x86_64-unknown-unknown -mstack-protector-guard-reg=fs %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-FS %s
// RUN: %clang -### -target x86_64-unknown-unknown -mstack-protector-guard-reg=gs %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-GS %s

// Invalid arch
// RUN: not %clang -target arm-eabi-c -mstack-protector-guard=tls %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-ARCH %s
// INVALID-ARCH: unsupported option '-mstack-protector-guard=tls' for target

// RUN: not %clang -target powerpc64le-linux-gnu -mstack-protector-guard-reg=fs %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-ARCH2 %s
// INVALID-ARCH2: unsupported option '-mstack-protector-guard-reg=fs' for target

// RUN: not %clang -target aarch64-linux-gnu -mstack-protector-guard-offset=10 %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-ARCH3 %s
// INVALID-ARCH3: unsupported option '-mstack-protector-guard-offset=10' for target

// Invalid option value
// RUN: not %clang -target x86_64-unknown-unknown -c -mstack-protector-guard-reg=cs %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-REG %s
// RUN: not %clang -target x86_64-unknown-unknown -c -mstack-protector-guard-reg=ds %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-REG %s

// CHECK-FS: "-cc1" {{.*}}"-mstack-protector-guard-reg=fs"
// CHECK-GS: "-cc1" {{.*}}"-mstack-protector-guard-reg=gs"
// INVALID-REG: error: invalid value {{.*}} in 'mstack-protector-guard-reg=','for X86, valid arguments to '-mstack-protector-guard-reg=' are:fs gs'

// RUN: %clang -### -target x86_64-unknown-unknown -mstack-protector-guard-offset=30 %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-OFFSET %s

// CHECK-OFFSET: "-cc1" {{.*}}"-mstack-protector-guard-offset=30"
