// Test that clang preincludes stdc-predef.h if we are using a libc that does
// not pre-include it, e.g. musl.

// Musl-based systems need this additional include.
//
// RUN: %clang %s -### -c 2>&1 \
// RUN: -target x86_64-unknown-linux-musl \
// RUN: --sysroot=%S/Inputs/stdc-predef \
// RUN: | FileCheck -check-prefix CHECK-CPP-FLAG %s

// GNU-based systems do not need this additional include.
//
// RUN: %clang %s -### -c 2>&1 \
// RUN: -target x86_64-unknown-linux-gnu \
// RUN: --sysroot=%S/Inputs/stdc-predef \
// RUN: | FileCheck --implicit-check-not "stdc-predef.h" %s

// Freestanding compilations do not need this additional include.
//
// RUN: %clang %s -### -c 2>&1 \
// RUN: -target x86_64-unknown-linux-musl -ffreestanding \
// RUN: --sysroot=%S/Inputs/stdc-predef \
// RUN: | FileCheck --implicit-check-not "stdc-predef.h" %s

// The last of -ffreestanding and -fhosted determines whether this additional
// include is needed.
//
// RUN: %clang %s -### -c 2>&1 \
// RUN: -target x86_64-unknown-linux-musl -ffreestanding -fhosted \
// RUN: --sysroot=%S/Inputs/stdc-predef \
// RUN: | FileCheck -check-prefix CHECK-CPP-FLAG %s
//
// RUN: %clang %s -### -c 2>&1 \
// RUN: -target x86_64-unknown-linux-musl -fhosted -ffreestanding \
// RUN: --sysroot=%S/Inputs/stdc-predef \
// RUN: | FileCheck --implicit-check-not "stdc-predef.h" %s

// Disabling standard include directories also disables this additional include.
//
// RUN: %clang %s -### -c 2>&1 \
// RUN: -target x86_64-unknown-linux-musl -nostdinc \
// RUN: --sysroot=%S/Inputs/stdc-predef \
// RUN: | FileCheck --implicit-check-not "stdc-predef.h" %s
//
// RUN: %clang %s -### -c 2>&1 \
// RUN: -target x86_64-unknown-linux-musl -nostdlibinc \
// RUN: --sysroot=%S/Inputs/stdc-predef \
// RUN: | FileCheck --implicit-check-not "stdc-predef.h" %s

// Because this behavior is implemented by adding preprocessor flags in the
// driver, disabling preprocessing means the include flag should not appear.
//
// RUN: %clang -x cpp-output %s -### -c 2>&1 \
// RUN: -target x86_64-unknown-linux-musl \
// RUN: --sysroot=%S/Inputs/stdc-predef \
// RUN: | FileCheck --implicit-check-not "stdc-predef.h" %s

// The automatic preinclude of stdc-predef.h should not occur if
// the source filename indicates a preprocessed file.
//
// RUN: %clang %S/Inputs/stdc-predef/preprocessed.i -### -c 2>&1 \
// RUN: --sysroot=%S/Inputs/stdc-predef \
// RUN: -target x86_64-unknown-linux-musl \
// RUN: | FileCheck --implicit-check-not "stdc-predef.h" %s

// This behavior should appear in all files clang accepts that use the C
// preprocessor, including C, C++ and Objective-C files, as long as the system
// uses musl.
//
// RUN: %clang -x objective-c %s -### -c 2>&1 \
// RUN: -target x86_64-unknown-linux-musl \
// RUN: --sysroot=%S/Inputs/stdc-predef \
// RUN: | FileCheck -check-prefix CHECK-CPP-FLAG %s

// If a musl-based system does not have this header, give an error at line 1.
//
// RUN: %clang %s -c -Xclang -verify=expected 2>&1 \
// RUN: -target x86_64-unknown-linux-musl \
// RUN: --sysroot=%S/Inputs/basic_linux_tree
// expected-error@1 {{'stdc-predef.h' file not found}}

// Check if the file is really included by macro.
//
// RUN: %clang %s -c -Xclang -verify=ok -DCHECK_DUMMY=1 \
// RUN: -target x86_64-unknown-linux-musl \
// RUN: --sysroot=%S/Inputs/stdc-predef
// ok-no-diagnostics

// CHECK-CPP-FLAG: "-include" "stdc-predef.h"
int i;
#if CHECK_DUMMY
#if !DUMMY_STDC_PREDEF
#error "Expected macro symbol DUMMY_STDC_PREDEF is not defined."
#endif
#endif
