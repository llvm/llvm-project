// REQUIRES: systemz-registered-target

// RUN: %clang -### -target s390x-ibm-zos -mzos-ppa1-name -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=EMIT-NAME %s
// EMIT-NAME: "-mzos-ppa1-name"

// RUN: %clang -### -target s390x-ibm-zos -mno-zos-ppa1-name -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NOT-EMIT-NAME %s
// NOT-EMIT-NAME: "-mno-zos-ppa1-name"

// RUN: not %clang -target systemz-unknown-elf -mzos-ppa1-name -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ERR %s
// RUN: not %clang -target systemz-unknown-elf -mno-zos-ppa1-name -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ERR %s
// ERR: error: unsupported option '-m{{.*}}zos-ppa1-name' for target 'systemz-unknown-elf'
