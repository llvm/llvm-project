// REQUIRES: shell, amdclang
// UNSUPPORTED: system-windows
//
// clang and links to amdclang are the same
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang --version 2>&1 > %t.clang.version
// RUN: ln -s amdclang %t/amdclang
// RUN: %t/amdclang --version 2>&1 > %t.amdclang.version
// RUN: diff %t.clang.version %t.amdclang.version
