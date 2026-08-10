// REQUIRES: system-windows
//
// RUN: rm -rf %t
// RUN: mkdir %t
//
// Note: %s and %S must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// The main test for clang-cl pch handling is cl-pch.cpp.  This file only checks
// a few things for .c inputs.

// /Yc with a .c file should build a c pch file.
// RUN: %clang_cl -Werror /Yc%S/Inputs/pchfile.h /FI%S/Inputs/pchfile.h /c /Fo%t/pchfile.obj /Fp%t/pchfile.pch -v -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC %s
// CHECK-YC: cc1{{.* .*}}-emit-pch
// CHECK-YC-SAME: -o
// CHECK-YC-SAME: pchfile.pch
// CHECK-YC-SAME: -x
// CHECK-YC-SAME: c-header

// But not if /TP changes the input language to C++.
// RUN: %clang_cl /TP -Werror /Yc%S/Inputs/pchfile.h /FI%S/Inputs/pchfile.h /c /Fo%t/pchfile.obj /Fp%t/pchfile.pch -v -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YCTP %s
// CHECK-YCTP: cc1{{.* .*}}-emit-pch
// CHECK-YCTP-SAME: -o
// CHECK-YCTP-SAME: pchfile.pch
// CHECK-YCTP-SAME: -x
// CHECK-YCTP-SAME: c++-header

// Except if a later /TC changes it back.
// RUN: %clang_cl -Werror /Yc%S/Inputs/pchfile.h /FI%S/Inputs/pchfile.h /c /Fo%t/pchfile.obj /Fp%t/pchfile.pch -v -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YCTPTC %s
// CHECK-YCTPTC: cc1{{.* .*}}-emit-pch
// CHECK-YCTPTC-SAME: -o
// CHECK-YCTPTC-SAME: pchfile.pch
// CHECK-YCTPTC-SAME: -x
// CHECK-YCTPTC-SAME: c-header

// Also check lower-case /Tp flag.
// RUN: %clang_cl -Werror /Tp%s /Yc%S/Inputs/pchfile.h /FI%S/Inputs/pchfile.h /c /Fo%t/pchfile.obj /Fp%t/pchfile.pch -v 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YCTp %s
// CHECK-YCTp: cc1{{.* .*}}-emit-pch
// CHECK-YCTp-SAME: -o
// CHECK-YCTp-SAME: pchfile.pch
// CHECK-YCTp-SAME: -x
// CHECK-YCTp-SAME: c++-header
