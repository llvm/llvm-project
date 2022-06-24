// Android, Cygwin and OpenBSD use emutls by default.
// Clang should pass -femulated-tls or -fno-emulated-tls to cc1 if they are used,
// and cc1 should set up EmulatedTLS and ExplicitEmulatedTLS to LLVM CodeGen.
//
// RUN: %clang -### -target arm-linux-androideabi %s 2>&1 \
// RUN: | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang -### -target arm-linux-gnu %s 2>&1 \
// RUN: | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang -### -target i686-pc-cygwin %s 2>&1 \
// RUN: | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang -### -target i686-pc-openbsd %s 2>&1 \
// RUN: | FileCheck -check-prefix=DEFAULT %s

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


// Default without -f[no-]emulated-tls, will be decided by the target triple.
// DEFAULT-NOT: "-cc1" {{.*}}"-femulated-tls"
// DEFAULT-NOT: "-cc1" {{.*}}"-fno-emulated-tls"

// Explicit and last -f[no-]emulated-tls flag will be passed to cc1.
// EMU: "-cc1" {{.*}}"-femulated-tls"
// EMU-NOT: "-cc1" {{.*}}"-fno-emulated-tls"

// NOEMU: "-cc1" {{.*}}"-fno-emulated-tls"
// NOEMU-NOT: "-cc1" {{.*}}"-femulated-tls"
