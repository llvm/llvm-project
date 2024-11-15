// RUN: %clang -### -target x86_64-unknown-unknown -mstack-protector-guard=tls %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-TLS %s
// RUN: %clang -### -target x86_64-unknown-unknown -mstack-protector-guard=global %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-GLOBAL %s
// RUN: not %clang -### --target=x86_64-unknown-unknown -mstack-protector-guard=local %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-VALUE %s

// CHECK-TLS: "-cc1" {{.*}}"-mstack-protector-guard=tls"
// CHECK-GLOBAL: "-cc1" {{.*}}"-mstack-protector-guard=global"
// INVALID-VALUE: error: invalid value 'local' in 'mstack-protector-guard=', expected one of: tls global

// RUN: %clang -### -target x86_64-unknown-unknown -mstack-protector-guard-reg=fs %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-FS %s
// RUN: %clang -### -target x86_64-unknown-unknown -mstack-protector-guard-reg=gs %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-GS %s
// RUN: %clang -### -target x86_64-unknown-unknown -mstack-protector-guard-symbol=sym %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-SYM %s

// Invalid arch
// RUN: not %clang -target powerpc64le-linux-gnu -mstack-protector-guard=tls %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-ARCH %s
// INVALID-ARCH: unsupported option '-mstack-protector-guard=tls' for target

// RUN: not %clang -target powerpc64le-linux-gnu -mstack-protector-guard-reg=fs %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-ARCH2 %s
// INVALID-ARCH2: unsupported option '-mstack-protector-guard-reg=fs' for target

// RUN: not %clang -target powerpc64le-linux-gnu -mstack-protector-guard-offset=10 %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-ARCH3 %s
// INVALID-ARCH3: unsupported option '-mstack-protector-guard-offset=10' for target

// Invalid option value
// RUN: not %clang -target x86_64-unknown-unknown -c -mstack-protector-guard-reg=cs %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-REG %s
// RUN: not %clang -target x86_64-unknown-unknown -c -mstack-protector-guard-reg=ds %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-REG %s
// RUN: not %clang -target x86_64-unknown-unknown -c -mstack-protector-guard-symbol=2s %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-SYM %s
// RUN: not %clang -target x86_64-unknown-unknown -c -mstack-protector-guard-symbol= %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-SYM %s

// CHECK-FS: "-cc1" {{.*}}"-mstack-protector-guard-reg=fs"
// CHECK-GS: "-cc1" {{.*}}"-mstack-protector-guard-reg=gs"
// INVALID-REG: error: invalid value {{.*}} in 'mstack-protector-guard-reg=', expected one of: fs gs
// CHECK-SYM: "-cc1" {{.*}}"-mstack-protector-guard-symbol=sym"
// INVALID-SYM: error: invalid argument 'mstack-protector-guard-symbol=' only allowed with 'legal symbol name'

// RUN: not %clang -target arm-eabi-c -mstack-protector-guard=tls %s 2>&1 | \
// RUN:   FileCheck -check-prefix=MISSING-OFFSET %s
// MISSING-OFFSET: error: '-mstack-protector-guard=tls' is used without '-mstack-protector-guard-offset', and there is no default

// RUN: not %clang -target arm-eabi-c -mstack-protector-guard-offset=1048576 %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-OFFSET %s
// INVALID-OFFSET: invalid integral value '1048576' in 'mstack-protector-guard-offset='

// RUN: not %clang -target arm-eabi-c -mstack-protector-guard=sysreg %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-VALUE2 %s
// INVALID-VALUE2: error: invalid value 'sysreg' in 'mstack-protector-guard=', expected one of: tls global

// RUN: not %clang -target thumbv6-eabi-c -mthumb -mstack-protector-guard=tls -mstack-protector-guard-offset=0 %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-ARCH4 %s
// INVALID-ARCH4: error: hardware TLS register is not supported for the thumbv6 sub-architecture

// RUN: not %clang -target thumbv7-eabi-c -mtp=soft -mstack-protector-guard=tls -mstack-protector-guard-offset=0 %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-TP %s
// INVALID-TP: error: invalid argument '-mtp=soft' not allowed with '-mstack-protector-guard=tls'

// RUN: %clang -### -target x86_64-unknown-unknown -mstack-protector-guard-offset=30 %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-OFFSET %s

// CHECK-OFFSET: "-cc1" {{.*}}"-mstack-protector-guard-offset=30"

// RUN: %clang -### -target aarch64-linux-gnu -mstack-protector-guard=sysreg \
// RUN:   -mstack-protector-guard-reg=sp_el0 \
// RUN:   -mstack-protector-guard-offset=0 %s 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-AARCH64 %s
// RUN: not %clang -### --target=aarch64-linux-gnu \
// RUN:   -mstack-protector-guard=tls %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-VALUE-AARCH64 %s
// RUN: not %clang -### --target=aarch64-linux-gnu -mstack-protector-guard=sysreg \
// RUN:   -mstack-protector-guard-reg=foo \
// RUN:   -mstack-protector-guard-offset=0 %s 2>&1 | \
// RUN: FileCheck -check-prefix=INVALID-REG-AARCH64 %s

// CHECK-AARCH64: "-cc1" {{.*}}"-mstack-protector-guard=sysreg" "-mstack-protector-guard-offset=0" "-mstack-protector-guard-reg=sp_el0"
// INVALID-VALUE-AARCH64: error: invalid value 'tls' in 'mstack-protector-guard=', expected one of: sysreg global
// INVALID-REG-AARCH64: error: invalid value 'foo' in 'mstack-protector-guard-reg='

// RUN: %clang -### -target riscv64-unknown-elf -mstack-protector-guard=tls -mstack-protector-guard-offset=24 -mstack-protector-guard-reg=tp %s 2>&1 | \
// RUN:   FileCheck -v -check-prefix=CHECK-TLS-RISCV %s
// RUN: %clang -### -target riscv64-unknown-elf -mstack-protector-guard=global %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-GLOBAL %s

// CHECK-TLS-RISCV: "-cc1" {{.*}}"-mstack-protector-guard=tls" "-mstack-protector-guard-offset=24" "-mstack-protector-guard-reg=tp"

// RUN: not %clang -target riscv64-unknown-elf -mstack-protector-guard=tls %s 2>&1 | \
// RUN:   FileCheck -check-prefix=MISSING-OFFSET %s

// RUN: not %clang -target riscv64-unknown-elf -mstack-protector-guard=sysreg %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-VALUE2 %s

// RUN: not %clang -target riscv64-unknown-elf -mstack-protector-guard=tls \
// RUN: -mstack-protector-guard-offset=20 -mstack-protector-guard-reg=sp %s 2>&1 | \
// RUN:   FileCheck -check-prefix=INVALID-REG-RISCV %s

// INVALID-REG-RISCV: error: invalid value 'sp' in 'mstack-protector-guard-reg=', expected one of: tp
