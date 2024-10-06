// REQUIRES: shell

// RUN: unset CLANG_NO_DEFAULT_CONFIG
// RUN: rm -rf %t && mkdir %t

//--- Major-versioned config files are used when targetting *-apple-darwin*
//
// RUN: mkdir -p %t/testbin
// RUN: ln -s %clang %t/testbin/x86_64-apple-darwin24.0.0-clang
// RUN: echo "-Werror" > %t/testbin/x86_64-apple-darwin24.cfg
// RUN: %t/testbin/x86_64-apple-darwin24.0.0-clang --config-system-dir= --config-user-dir= -c -no-canonical-prefixes -### %s 2>&1 | FileCheck %s -check-prefix CHECK-MAJOR-VERSIONED
//
// CHECK-MAJOR-VERSIONED: Configuration file: {{.*}}/testbin/x86_64-apple-darwin24.cfg
// CHECK-MAJOR-VERSIONED: -Werror

//--- Unversioned config files are used when targetting *-apple-darwin*
//
// RUN: mkdir -p %t/testbin
// RUN: ln -s %clang %t/testbin/arm64-apple-darwin23.1.2-clang
// RUN: echo "-Werror" > %t/testbin/arm64-apple-darwin.cfg
// RUN: %t/testbin/arm64-apple-darwin23.1.2-clang --config-system-dir= --config-user-dir= -c -no-canonical-prefixes -### %s 2>&1 | FileCheck %s -check-prefix CHECK-UNVERSIONED
//
// CHECK-UNVERSIONED: Configuration file: {{.*}}/testbin/arm64-apple-darwin.cfg
// CHECK-UNVERSIONED: -Werror
