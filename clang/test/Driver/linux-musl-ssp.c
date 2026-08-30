// Check that on musl the driver links libssp_nonshared.a when stack
// protection is enabled and the sysroot provides the library.

// RUN: %clang -### --target=i686-unknown-linux-musl \
// RUN:   --sysroot=%S/Inputs/musl_ssp_tree -fstack-protector-strong %s 2>&1 \
// RUN:   | FileCheck --check-prefix=SSP %s
// SSP: "-lc" "-lssp_nonshared"

// Not with stack protection disabled (last flag wins).
// RUN: %clang -### --target=i686-unknown-linux-musl \
// RUN:   --sysroot=%S/Inputs/musl_ssp_tree -fstack-protector-strong \
// RUN:   -fno-stack-protector %s 2>&1 | FileCheck --check-prefix=NOSSP %s
// NOSSP-NOT: "-lssp_nonshared"

// Not without any stack protector flag.
// RUN: %clang -### --target=i686-unknown-linux-musl \
// RUN:   --sysroot=%S/Inputs/musl_ssp_tree %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOSSP %s

// Not on glibc: libc_nonshared.a is linked via the libc.so linker script.
// RUN: %clang -### --target=i686-unknown-linux-gnu \
// RUN:   --sysroot=%S/Inputs/musl_ssp_tree -fstack-protector-strong %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOSSP %s

// Not when the sysroot does not provide the library.
// RUN: %clang -### --target=i686-unknown-linux-musl \
// RUN:   --sysroot=%S/Inputs/basic_linux_tree -fstack-protector-strong %s \
// RUN:   2>&1 | FileCheck --check-prefix=NOSSP %s
