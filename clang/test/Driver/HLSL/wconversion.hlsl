// RUN: %clang_dxc -T lib_6_7 %s -### %s 2>&1 | FileCheck %s --check-prefixes=CONV
// RUN: %clang_dxc -T lib_6_7 -Wno-conversion %s -### %s 2>&1 | FileCheck %s --check-prefixes=NOCONV

// make sure we generate -Wconversion by default
// CONV: "-Wconversion"
// make sure -Wno-conversion still works
// NOCONV: "-Wno-conversion"
