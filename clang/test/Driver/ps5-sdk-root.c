/// (Essentially identical to ps4-sdk-root.c except for the target.)

/// Check that PS5 clang doesn't report a warning message when locating
/// system header files (either by looking at the value of SCE_PROSPERO_SDK_DIR
/// or relative to the location of the compiler driver), if "-nostdinc",
/// "--sysroot" or "-isysroot" option is specified on the command line.
/// Otherwise, check that PS5 clang reports a warning.

// Check that PS5 clang doesn't report a warning message when locating
// system libraries (either by looking at the value of SCE_PROSPERO_SDK_DIR
// or relative to the location of the compiler driver), if "-c", "-S", "-E",
// "--sysroot", "-nostdlib" or "-nodefaultlibs" option is specified on
// the command line.
// Otherwise, check that PS5 clang reports a warning.

// Setting up SCE_PROSPERO_SDK_DIR to existing location, which is not a PS4 SDK.
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=WARN-SYS-HEADERS -check-prefix=WARN-SYS-LIBS -check-prefix=NO-WARN %s

// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -c -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=WARN-SYS-HEADERS -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -S -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=WARN-SYS-HEADERS -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -E -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=WARN-SYS-HEADERS -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -emit-ast -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=WARN-SYS-HEADERS -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -isysroot foo -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=WARN-ISYSROOT -check-prefix=WARN-SYS-LIBS -check-prefix=NO-WARN %s

// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -c -nostdinc -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -S -nostdinc -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -E -nostdinc -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -emit-ast -nostdinc -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s

// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -c --sysroot=foo/ -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -S --sysroot=foo/ -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -E --sysroot=foo/ -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -emit-ast --sysroot=foo/ -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s

// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -c -isysroot foo -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=WARN-ISYSROOT -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -S -isysroot foo -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=WARN-ISYSROOT -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -E -isysroot foo -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=WARN-ISYSROOT -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -emit-ast -isysroot foo -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=WARN-ISYSROOT -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### --sysroot=foo/ -isysroot foo -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=WARN-ISYSROOT -check-prefix=NO-WARN %s

// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -nostdlib -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=WARN-SYS-HEADERS -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang -Winvalid-or-nonexistent-directory -### -nodefaultlibs -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=WARN-SYS-HEADERS -check-prefix=NO-WARN %s

// NO-WARN-NOT: {{warning:|error:}}
// WARN-SYS-HEADERS: warning: unable to find PS5 system headers directory
// WARN-ISYSROOT: warning: no such sysroot directory: 'foo'
// WARN-SYS-LIBS: warning: unable to find PS5 system libraries directory
// NO-WARN-NOT: {{warning:|error:}}
