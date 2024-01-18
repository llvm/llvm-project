// The default ABI is aapcs
// RUN: %clang -target arm--- %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS %s
// RUN: %clang -target armeb--- %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS %s
// RUN: %clang -target thumb--- %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS %s
// RUN: %clang -target thumbeb--- %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS %s

// MachO targets default to apcs-gnu, except for m-class processors
// RUN: %clang -target arm--darwin- -arch armv7s %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-APCS-GNU %s
// RUN: %clang -target thumb--darwin- -arch armv7s %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-APCS-GNU %s
// RUN: %clang -target thumb--darwin- -arch armv7m %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS %s

// Windows targets default to AAPCS, regardless of environment
// RUN: %clang -target arm--windows-gnueabi %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS %s

// NetBSD defaults to apcs-gnu, but can also use aapcs
// RUN: %clang -target arm--netbsd- %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-APCS-GNU %s
// RUN: %clang -target arm--netbsd-eabi %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS %s
// RUN: %clang -target arm--netbsd-eabihf %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS %s

// FreeBSD / OpenBSD default to aapcs-linux
// RUN: %clang -target arm--freebsd- %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS-LINUX %s
// RUN: %clang -target arm--openbsd- %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS-LINUX %s
// RUN: %clang -target arm--haiku- %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS-LINUX %s

// Otherwise, ABI is selected based on environment
// RUN: %clang -target arm---android %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS-LINUX %s
// RUN: %clang -target arm---gnueabi %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS-LINUX %s
// RUN: %clang -target arm---gnueabihf %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS-LINUX %s
// RUN: %clang -target arm---musleabi %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS-LINUX %s
// RUN: %clang -target arm---musleabihf %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS-LINUX %s
// RUN: %clang -target arm---eabi %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS %s
// RUN: %clang -target arm---eabihf %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS %s

// ABI can be overridden by the -mabi= option
// RUN: %clang -target arm---eabi -mabi=apcs-gnu %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-APCS-GNU %s
// RUN: %clang -target arm---gnueabi -mabi=aapcs %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS %s
// RUN: %clang -target arm---eabi -mabi=aapcs-linux %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AAPCS-LINUX %s

// CHECK-APCS-GNU: "-target-abi" "apcs-gnu"
// CHECK-AAPCS: "-target-abi" "aapcs"
// CHECK-AAPCS-LINUX: "-target-abi" "aapcs-linux"

// RUN: %clang --target=arm---gnueabi -mabi=aapcs -x assembler %s -### -o /dev/null 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ASM %s
// RUN: %clang --target=arm---gnueabi -mabi=aapcs -x assembler %s -### -o /dev/null -fno-integrated-as 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ASM %s

/// The combination -x assember & -mabi is not implemented, but for GCC compatibility we accept with a warning.
// CHECK-ASM: warning: argument unused during compilation: '-mabi={{.*}}'
