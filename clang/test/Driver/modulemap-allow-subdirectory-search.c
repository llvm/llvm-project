// RUN: rm -rf %t

// Check that searching for module maps in subdirectories of search paths is
// off by default on all platforms and only enabled when explicitly requested.

// RUN: %clang -target x86_64-apple-macos10.13 -fmodules %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-SUBDIRECTORIES %s
// RUN: %clang -target i386-unknown-linux -fmodules %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-SUBDIRECTORIES %s
// RUN: %clang -target x86_64-pc-windows-msvc -fmodules %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-SUBDIRECTORIES %s
// RUN: %clang -target x86_64-apple-macos10.13 -fmodules %s -fmodulemap-allow-subdirectory-search \
// RUN:   -fno-modulemap-allow-subdirectory-search -### 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-SUBDIRECTORIES %s

// RUN: %clang -target x86_64-apple-macos10.13 -fmodules %s -fmodulemap-allow-subdirectory-search -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SEARCH-SUBDIRECTORIES %s
// RUN: %clang -target i386-unknown-linux -fmodules %s -fmodulemap-allow-subdirectory-search -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SEARCH-SUBDIRECTORIES %s

// NO-SUBDIRECTORIES-NOT: "-fmodulemap-allow-subdirectory-search"
// NO-SUBDIRECTORIES-NOT: "-fno-modulemap-allow-subdirectory-search"
// SEARCH-SUBDIRECTORIES: "-fmodulemap-allow-subdirectory-search"
// SEARCH-SUBDIRECTORIES-NOT: "-fno-modulemap-allow-subdirectory-search"
