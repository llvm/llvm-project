// RUN: %clang_dxc -T lib_6_7 %s -### %s 2>&1 | FileCheck %s --check-prefixes=CONV
// RUN: %clang_dxc -T lib_6_7 -Wno-conversion -Wno-vector-conversion -Wno-matrix-conversion %s -### %s 2>&1 | FileCheck %s --check-prefixes=NOCONV

// make sure we generate -Wconversion by default
// CONV: "-Wconversion"
// make sure -Wno-conversion still works
// NOCONV: "-Wno-conversion"

// make sure we generate -Wvector-conversion by default
// CONV: "-Wvector-conversion"
// make sure -Wno-vector-conversion still works
// NOCONV: "-Wno-vector-conversion"

// make sure we generate -Wmatrix-conversion by default
// CONV: "-Wmatrix-conversion"
// make sure -Wno-matrix-conversion still works
// NOCONV: "-Wno-matrix-conversion"
