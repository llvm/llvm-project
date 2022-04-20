// RUN: %clang -target i386 -### -no-integrated-as -c %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NOIAS

// NOIAS: -no-integrated-as

// RUN: %clang -target i386 -### -integrated-as -c %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix IAS

// IAS-NOT: -no-integrated-as

// RUN: %clang -target i386 -### -c %s 2>&1 | FileCheck %s -check-prefix DEFAULT
// RUN: %clang -target msp430 -### -c %s 2>&1 | FileCheck %s -check-prefix DEFAULT

// DEFAULT-NOT: -no-integrated-as

/// -c omitted on purpose, this used to crash.
// RUN: %clang -target x86_64-sie-ps5 -### -no-integrated-as %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix PS5
// RUN: %clang -target x86_64-sie-ps5 -### -no-integrated-as -x assembler %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix PS5
// PS5: there is no external assembler that can be used on this platform
