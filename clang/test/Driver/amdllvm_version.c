// REQUIRES: amdclang
//
// clang and amdclang are the same
// RUN: %clang --version 2>&1 > %t.clang.version
// RUN: amdclang --version 2>&1 > %t.amdclang.version
// RUN: diff %t.clang.version %t.amdclang.version
