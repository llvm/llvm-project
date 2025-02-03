/// PS4 clang emits warnings when SDK headers (<SDKROOT>/target/include/) or
/// libraries (<SDKROOT>/target/lib/) are missing. If the the user takes control
/// of header search paths, the existence check for <SDKROOT>/target/include is
/// skipped.
///
/// User control of header search is assumed if `--sysroot`, `-isysroot`,
/// `-nostdinc` or `-nostdlibinc` is supplied.
///
/// Warnings are emitted if a specified `-isysroot` or `--sysroot` does not
/// exist.
///
/// The default <SDKROOT> for both headers and libraries is taken from the
/// SCE_ORBIS_SDK_DIR environment variable.

// RUN: echo "-### -Winvalid-or-nonexistent-directory -target x86_64-scei-ps4" > %t.rsp

/// If SDK headers and/or libraries are found, associated warnings are absent.
// RUN: rm -rf %t.inconly && mkdir -p %t.inconly/target/include
// RUN: env SCE_ORBIS_SDK_DIR=%t.inconly %clang @%t.rsp %s 2>&1 | FileCheck -check-prefixes=WARN-SYS-LIBS,NO-WARN %s

// RUN: rm -rf %t.libonly && mkdir -p %t.libonly/target/lib
// RUN: env SCE_ORBIS_SDK_DIR=%t.libonly %clang @%t.rsp %s 2>&1 | FileCheck -check-prefixes=WARN-SYS-HEADERS,NO-WARN %s

// RUN: rm -rf %t.both && mkdir -p %t.both/target/lib && mkdir %t.both/target/include
// RUN: env SCE_ORBIS_SDK_DIR=%t.both %clang @%t.rsp %s 2>&1 | FileCheck -check-prefix=NO-WARN %s

/// In the following invocations, SCE_ORBIS_SDK_DIR is set to an existing
/// location where SDK headers and libraries are absent.

/// When compiling and linking, we should see a warnings about both missing
/// headers and libraries.
// RUN: env SCE_ORBIS_SDK_DIR=.. %clang @%t.rsp %s 2>&1 | FileCheck -check-prefixes=WARN-SYS-HEADERS,WARN-SYS-LIBS,NO-WARN %s

/// If `-c`, `-S`, `-E` or `-emit-ast` is supplied, the existence check for SDK
/// libraries is skipped because no linking will be performed. We only expect
/// warnings about missing headers.
// RUN: env SCE_ORBIS_SDK_DIR=.. %clang @%t.rsp %s -c 2>&1 | FileCheck -check-prefixes=WARN-SYS-HEADERS,NO-WARN %s
// RUN: env SCE_ORBIS_SDK_DIR=.. %clang @%t.rsp %s -S 2>&1 | FileCheck -check-prefixes=WARN-SYS-HEADERS,NO-WARN %s
// RUN: env SCE_ORBIS_SDK_DIR=.. %clang @%t.rsp %s -E 2>&1 | FileCheck -check-prefixes=WARN-SYS-HEADERS,NO-WARN %s
// RUN: env SCE_ORBIS_SDK_DIR=.. %clang @%t.rsp %s -emit-ast 2>&1 | FileCheck -check-prefixes=WARN-SYS-HEADERS,NO-WARN %s

/// If the user takes control of include paths, the existence check for headers
/// is not performed.
// RUN: env SCE_ORBIS_SDK_DIR=.. %clang @%t.rsp %s -c -nostdinc 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_ORBIS_SDK_DIR=.. %clang @%t.rsp %s -c -nostdlibinc 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_ORBIS_SDK_DIR=.. %clang @%t.rsp %s -c -isysroot . 2>&1 | FileCheck -check-prefixes=NO-WARN %s
// RUN: env SCE_ORBIS_SDK_DIR=.. %clang @%t.rsp %s -c --sysroot=. 2>&1 | FileCheck -check-prefixes=NO-WARN %s

/// --sysroot disables the existence check for headers. The check for libraries
/// remains.
// RUN: env SCE_ORBIS_SDK_DIR=.. %clang @%t.rsp %s --sysroot=. 2>&1 | FileCheck -check-prefixes=WARN-SYS-LIBS,NO-WARN %s

/// -isysroot overrides --sysroot for header search, but not library search.
// RUN: env SCE_ORBIS_SDK_DIR=.. %clang @%t.rsp %s -isysroot . --sysroot=%t.inconly 2>&1 | FileCheck -check-prefixes=ISYSTEM,WARN-SYS-LIBS,NO-WARN %s
// RUN: env SCE_ORBIS_SDK_DIR=.. %clang @%t.rsp %s --sysroot=%t.inconly -isysroot . 2>&1 | FileCheck -check-prefixes=ISYSTEM,WARN-SYS-LIBS,NO-WARN %s

/// Warnings are emitted if non-existent --sysroot/-isysroot are supplied.
// RUN: env SCE_ORBIS_SDK_DIR=.. %clang @%t.rsp %s --sysroot=foo -isysroot %t.both 2>&1 | FileCheck -check-prefixes=WARN-SYSROOT,WARN-SYS-LIBS,NO-WARN %s
// RUN: env SCE_ORBIS_SDK_DIR=.. %clang @%t.rsp %s -isysroot foo --sysroot=%t.both 2>&1 | FileCheck -check-prefixes=WARN-SYSROOT,NO-WARN %s
// RUN: env SCE_ORBIS_SDK_DIR=.. %clang @%t.rsp %s --sysroot=foo -isysroot bar 2>&1 | FileCheck -check-prefixes=WARN-SYSROOT,WARN-SYSROOT2,WARN-SYS-LIBS,NO-WARN %s

// NO-WARN-NOT: {{warning:|error:}}
// WARN-SYSROOT: warning: no such sysroot directory: 'foo'
// WARN-SYSROOT2: warning: no such sysroot directory: 'bar'
// WARN-SYS-LIBS: warning: unable to find PS4 system libraries directory
// WARN-SYS-HEADERS: warning: unable to find PS4 system headers directory
// NO-WARN-NOT: {{warning:|error:}}
// ISYSTEM: "-cc1"{{.*}}"-internal-externc-isystem" "./target/include"
