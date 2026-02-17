// REQUIRES: shell
// REQUIRES: x86-registered-target

// RUN: unset CLANG_NO_DEFAULT_CONFIG
// RUN: rm -rf %t && mkdir %t

// --- Clang started via an alternative prefix is preferred over "real" triple.
//
// RUN: mkdir %t/testdmode
// RUN: ln -s %clang %t/testdmode/x86_64-w64-windows-gnu-clang-g++
// RUN: ln -s %clang %t/testdmode/x86_64-w64-windows-gnu-clang
// RUN: ln -s %clang %t/testdmode/x86_64-w64-mingw32-clang-g++
// RUN: ln -s %clang %t/testdmode/x86_64-w64-mingw32-clang
// RUN: touch %t/testdmode/x86_64-w64-windows-gnu-clang++.cfg
// RUN: touch %t/testdmode/x86_64-w64-windows-gnu-clang-g++.cfg
// RUN: touch %t/testdmode/x86_64-w64-windows-gnu-clang.cfg
// RUN: touch %t/testdmode/x86_64-w64-windows-gnu.cfg
// RUN: touch %t/testdmode/x86_64-w64-mingw32-clang++.cfg
// RUN: touch %t/testdmode/x86_64-w64-mingw32-clang-g++.cfg
// RUN: touch %t/testdmode/x86_64-w64-mingw32-clang.cfg
// RUN: touch %t/testdmode/x86_64-w64-mingw32.cfg
// RUN: touch %t/testdmode/clang++.cfg
// RUN: touch %t/testdmode/clang-g++.cfg
// RUN: touch %t/testdmode/clang.cfg
// RUN: %t/testdmode/x86_64-w64-mingw32-clang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1-ALT --implicit-check-not 'Configuration file:'
//
// FULL1-ALT: Configuration file: {{.*}}/testdmode/x86_64-w64-mingw32-clang++.cfg

// --- Test fallback to real triple.
//
// RUN: rm %t/testdmode/x86_64-w64-mingw32-clang++.cfg
// RUN: %t/testdmode/x86_64-w64-mingw32-clang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL2-ALT --implicit-check-not 'Configuration file:'
//
// FULL2-ALT: Configuration file: {{.*}}/testdmode/x86_64-w64-windows-gnu-clang++.cfg

// --- Test fallback to x86_64-w64-mingw32-clang-g++.cfg.
//
// RUN: rm %t/testdmode/x86_64-w64-windows-gnu-clang++.cfg
// RUN: %t/testdmode/x86_64-w64-mingw32-clang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL3-ALT --implicit-check-not 'Configuration file:'
//
// FULL3-ALT: Configuration file: {{.*}}/testdmode/x86_64-w64-mingw32-clang-g++.cfg

// --- Test fallback to real triple.
//
// RUN: rm %t/testdmode/x86_64-w64-mingw32-clang-g++.cfg
// RUN: %t/testdmode/x86_64-w64-mingw32-clang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL4-ALT --implicit-check-not 'Configuration file:'
//
// FULL4-ALT: Configuration file: {{.*}}/testdmode/x86_64-w64-windows-gnu-clang-g++.cfg

//--- Test fallback to x86_64-w64-mingw32.cfg + clang++.cfg.
//
// RUN: rm %t/testdmode/x86_64-w64-windows-gnu-clang-g++.cfg
// RUN: %t/testdmode/x86_64-w64-mingw32-clang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix ALT-FALLBACK1 --implicit-check-not 'Configuration file:'
//
// ALT-FALLBACK1: Configuration file: {{.*}}/testdmode/clang++.cfg
// ALT-FALLBACK1: Configuration file: {{.*}}/testdmode/x86_64-w64-mingw32.cfg

// --- Test fallback to real triple.
//
// RUN: rm %t/testdmode/x86_64-w64-mingw32.cfg
// RUN: %t/testdmode/x86_64-w64-mingw32-clang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix ALT-FALLBACK2 --implicit-check-not 'Configuration file:'
//
// ALT-FALLBACK2: Configuration file: {{.*}}/testdmode/clang++.cfg
// ALT-FALLBACK2: Configuration file: {{.*}}/testdmode/x86_64-w64-windows-gnu.cfg
