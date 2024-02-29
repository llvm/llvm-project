// Android, Cygwin and OpenBSD use emutls by default.
// Clang should pass -femulated-tls to cc1 if they are used,
// and cc1 should set up EmulatedTLS to LLVM CodeGen.
//
// RUN: %clang -### --target=arm-linux-androideabi %s 2>&1 \
// RUN:   | FileCheck -check-prefix=EMU %s
// RUN: %clang -### --target=arm-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NOEMU %s
// RUN: %clang -### --target=i686-pc-cygwin %s 2>&1 \
// RUN:   | FileCheck -check-prefix=EMU %s
// RUN: %clang -### --target=i686-pc-openbsd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=EMU %s

// RUN: %clang -### -target arm-linux-androideabi -fno-emulated-tls -femulated-tls %s 2>&1 \
// RUN: | FileCheck -check-prefix=EMU %s
// RUN: %clang -### -target arm-linux-gnu %s -fno-emulated-tls -femulated-tls 2>&1 \
// RUN: | FileCheck -check-prefix=EMU %s
// RUN: %clang -### -target i686-pc-cygwin %s -fno-emulated-tls -femulated-tls 2>&1 \
// RUN: | FileCheck -check-prefix=EMU %s
// RUN: %clang -### -target i686-pc-openbsd %s -fno-emulated-tls -femulated-tls 2>&1 \
// RUN: | FileCheck -check-prefix=EMU %s

// RUN: %clang -### -target arm-linux-androideabi -femulated-tls -fno-emulated-tls %s 2>&1 \
// RUN: | FileCheck -check-prefix=NOEMU %s
// RUN: %clang -### -target arm-linux-gnu %s -femulated-tls -fno-emulated-tls 2>&1 \
// RUN: | FileCheck -check-prefix=NOEMU %s
// RUN: %clang -### -target i686-pc-cygwin %s -femulated-tls -fno-emulated-tls 2>&1 \
// RUN: | FileCheck -check-prefix=NOEMU %s
// RUN: %clang -### -target i686-pc-openbsd %s -femulated-tls -fno-emulated-tls 2>&1 \
// RUN: | FileCheck -check-prefix=NOEMU %s

// Test that when lto is used any -emualted-tls flags are passed to the linker
// LINUX and Android have different defaults for EmulatedTLS
// RUN: %clang -### -flto --target=riscv64-linux -fno-emulated-tls %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=LTO_NOEMUTLS
// RUN: %clang -### -flto --target=riscv64-linux-android10000 -femulated-tls %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=LTO_EMUTLS
// RUN: %clang -### -flto --target=riscv64-linux -femulated-tls %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=LTO_EMUTLS
// RUN: %clang -### -flto --target=riscv64-linux-android10000 -fno-emulated-tls %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=LTO_NOEMUTLS
// RUN: %clang -### -flto --target=amd64-unknown-openbsd %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=LTO_EMUTLS

// Default without -f[no-]emulated-tls, will be decided by the target triple.
// DEFAULT-NOT: "-cc1" {{.*}}"-femulated-tls"

// EMU:      "-cc1"
// EMU-SAME: "-femulated-tls"

// NOEMU:     "-cc1"
// NOEMU-NOT: "-femulated-tls"

// LTO related checks
// LTO_NOEMUTLS-NOT: "-plugin-opt=-emulated-tls"

// LTO_EMUTLS: "-plugin-opt=-emulated-tls"
