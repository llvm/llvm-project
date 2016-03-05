// Note: %s and %S must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// The main test for clang-cl pch handling is cl-pch.cpp.  This file only checks
// a few things for .c inputs.

// /Yc with a .c file should build a c pch file.
// RUN: %clang_cl -Werror /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC %s
// CHECK-YC: cc1
// CHECK-YC: -emit-pch
// CHECK-YC: -o
// CHECK-YC: pchfile.pch
// CHECK-YC: -x
// CHECK-YC: "c"

// But not if /TP changes the input language to C++.
// RUN: %clang_cl /TP -Werror /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YCTP %s
// CHECK-YCTP: cc1
// CHECK-YCTP: -emit-pch
// CHECK-YCTP: -o
// CHECK-YCTP: pchfile.pch
// CHECK-YCTP: -x
// CHECK-YCTP: "c++"

// Except if a later /TC changes it back.
// RUN: %clang_cl -Werror /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YCTPTC %s
// CHECK-YCTPTC: cc1
// CHECK-YCTPTC: -emit-pch
// CHECK-YCTPTC: -o
// CHECK-YCTPTC: pchfile.pch
// CHECK-YCTPTC: -x
// CHECK-YCTPTC: "c"

// Also check lower-case /Tp flag.
// RUN: %clang_cl -Werror /Tp%s /Ycpchfile.h /FIpchfile.h /c -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YCTp %s
// CHECK-YCTp: cc1
// CHECK-YCTp: -emit-pch
// CHECK-YCTp: -o
// CHECK-YCTp: pchfile.pch
// CHECK-YCTp: -x
// CHECK-YCTp: "c++"
