// Test that clang preincludes stdc-predef.h for musl Linux targets and
// glibc Linux sysroots that provide the header.

// Musl-based systems need this additional include.
// RUN: %clang %s -### -c --target=x86_64-unknown-linux-musl --sysroot=%S/Inputs/stdc-predef 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CPP-FLAG %s

// GNU-based sysroots with stdc-predef.h also need this additional include.
// RUN: %clang %s -### -c --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/stdc-predef 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CPP-FLAG %s

// GNU-based sysroots without stdc-predef.h should not inject a host header.
// RUN: %clang %s -### -c --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --implicit-check-not "stdc-predef.h" %s

// Freestanding compilations do not need this additional include.
// RUN: %clang %s -### -c --target=x86_64-unknown-linux-musl -ffreestanding --sysroot=%S/Inputs/stdc-predef 2>&1 \
// RUN:   | FileCheck --implicit-check-not "stdc-predef.h" %s

// The last of -ffreestanding and -fhosted determines whether this additional
// include is needed.
// RUN: %clang %s -### -c --target=x86_64-unknown-linux-musl -ffreestanding -fhosted --sysroot=%S/Inputs/stdc-predef 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CPP-FLAG %s
// RUN: %clang %s -### -c --target=x86_64-unknown-linux-musl -fhosted -ffreestanding --sysroot=%S/Inputs/stdc-predef 2>&1 \
// RUN:   | FileCheck --implicit-check-not "stdc-predef.h" %s

// Disabling standard include directories also disables this additional include.
// RUN: %clang %s -### -c --target=x86_64-unknown-linux-musl -nostdinc --sysroot=%S/Inputs/stdc-predef 2>&1 \
// RUN:   | FileCheck --implicit-check-not "stdc-predef.h" %s
// RUN: %clang %s -### -c --target=x86_64-unknown-linux-musl -nostdlibinc --sysroot=%S/Inputs/stdc-predef 2>&1 \
// RUN:   | FileCheck --implicit-check-not "stdc-predef.h" %s

// Because this behavior is implemented by adding preprocessor flags in the
// driver, disabling preprocessing means the include flag should not appear.
// RUN: %clang -x cpp-output %s -### -c --target=x86_64-unknown-linux-musl --sysroot=%S/Inputs/stdc-predef 2>&1 \
// RUN:   | FileCheck --implicit-check-not "stdc-predef.h" %s

// The automatic preinclude of stdc-predef.h should not occur if
// the source filename indicates a preprocessed file.
// RUN: %clang %S/Inputs/stdc-predef/preprocessed.i -### -c --target=x86_64-unknown-linux-musl --sysroot=%S/Inputs/stdc-predef 2>&1 \
// RUN:   | FileCheck --implicit-check-not "stdc-predef.h" %s

// This behavior should appear in all files clang accepts that use the C
// preprocessor, including C, C++ and Objective-C files, as long as the system
// uses musl or a glibc sysroot that provides the header.
// RUN: %clang -x objective-c %s -### -c --target=x86_64-unknown-linux-musl --sysroot=%S/Inputs/stdc-predef 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CPP-FLAG %s

// If a target system does not have this header, give an error at line 1.
// RUN: not %clang %s -fsyntax-only --target=x86_64-unknown-linux-musl --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MISSING %s

// Check if the file is really included by macro.
// RUN: %clang %s -E -dM --target=x86_64-unknown-linux-musl --sysroot=%S/Inputs/stdc-predef 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MACRO %s

// CHECK-CPP-FLAG: "-include" "stdc-predef.h"
// CHECK-MISSING: 'stdc-predef.h' file not found
// CHECK-MACRO: #define DUMMY_STDC_PREDEF 1
int i;
