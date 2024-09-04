/// (Essentially identical to ps4-sdk-root.c except for the target.)

/// PS5 clang emits warnings when SDK headers or libraries are missing, or if a
/// specified `-isysroot` or `--sysroot` does not exist.
///
/// If the user takes control of header or library search, the respective
/// existence check is skipped. User control of header search is assumed if
/// `-isysroot`, `-nostdinc` or `-nostdlibinc` is supplied. User control of
/// library search is assumed if `--sysroot` is supplied.
///
/// The default sysroot for both headers and libraries is taken from the
/// SCE_PROSPERO_SDK_DIR environment variable.

// RUN: echo "-### -Winvalid-or-nonexistent-directory -target x86_64-sie-ps5" > %t.rsp

/// If SDK headers and/or libraries are found, associated warnings are absent.
// RUN: rm -rf %t.inconly && mkdir -p %t.inconly/target/include
// RUN: env SCE_PROSPERO_SDK_DIR=%t.inconly %clang @%t.rsp %s 2>&1 | FileCheck -check-prefixes=WARN-SYS-LIBS,NO-WARN %s

// RUN: rm -rf %t.libonly && mkdir -p %t.libonly/target/lib
// RUN: env SCE_PROSPERO_SDK_DIR=%t.libonly %clang @%t.rsp %s 2>&1 | FileCheck -check-prefixes=WARN-SYS-HEADERS,NO-WARN %s

// RUN: rm -rf %t.both && mkdir -p %t.both/target/lib && mkdir %t.both/target/include
// RUN: env SCE_PROSPERO_SDK_DIR=%t.both %clang @%t.rsp %s 2>&1 | FileCheck -check-prefix=NO-WARN %s

/// In the following invocations, SCE_PROSPERO_SDK_DIR is set to an existing
/// location where SDK headers and libraries are absent.

/// When compiling and linking, we should see a warnings about both missing
/// headers and libraries.
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang @%t.rsp %s 2>&1 | FileCheck -check-prefixes=WARN-SYS-HEADERS,WARN-SYS-LIBS,NO-WARN %s

/// If `-c`, `-S`, `-E` or `-emit-ast` is supplied, the existence check for SDK
/// libraries is skipped because no linking will be performed. We only expect
/// warnings about missing headers.
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang @%t.rsp %s -c 2>&1 | FileCheck -check-prefixes=WARN-SYS-HEADERS,NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang @%t.rsp %s -S 2>&1 | FileCheck -check-prefixes=WARN-SYS-HEADERS,NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang @%t.rsp %s -E 2>&1 | FileCheck -check-prefixes=WARN-SYS-HEADERS,NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang @%t.rsp %s -emit-ast 2>&1 | FileCheck -check-prefixes=WARN-SYS-HEADERS,NO-WARN %s

/// If the user takes control of include paths, the existence check for headers
/// is not performed.
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang @%t.rsp %s -c -nostdinc 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang @%t.rsp %s -c -nostdlibinc 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang @%t.rsp %s -c -isysroot . 2>&1 | FileCheck -check-prefixes=NO-WARN %s

/// --sysroot disables the existence check for libraries.
// RUN: touch %t.o
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang @%t.rsp %s --sysroot=. 2>&1 | FileCheck -check-prefixes=WARN-SYS-HEADERS,NO-WARN %s

/// Warnings are emitted if non-existent -isysroot/--sysroot are supplied.
// RUN: env SCE_PROSPERO_SDK_DIR=.. %clang @%t.rsp %s -isysroot foo --sysroot=bar 2>&1 | FileCheck -check-prefixes=WARN-SYSROOT,WARN-SYSROOT2,NO-WARN %s

// NO-WARN-NOT: {{warning:|error:}}
// WARN-SYS-HEADERS: warning: unable to find PS5 system headers directory
// WARN-SYSROOT: warning: no such sysroot directory: 'foo'
// WARN-SYSROOT2: warning: no such sysroot directory: 'bar'
// WARN-SYS-LIBS: warning: unable to find PS5 system libraries directory
// NO-WARN-NOT: {{warning:|error:}}
