// RUN: %clang -target mipsel-unknown-linux -mnan=2008 -march=mips2 -emit-llvm -S %s -o - 2>%t | FileCheck -check-prefix=CHECK-NANLEGACY %s
// RUN: FileCheck -check-prefix=CHECK-MIPS2 %s < %t
//
// RUN: %clang -target mips64el-unknown-linux -mnan=2008 -march=mips3 -emit-llvm -S %s -o - 2>%t | FileCheck -check-prefix=CHECK-NANLEGACY %s
// RUN: FileCheck -check-prefix=CHECK-MIPS3 %s < %t
//
// RUN: %clang -target mips64el-unknown-linux -mnan=2008 -march=mips4 -emit-llvm -S %s -o - 2>%t | FileCheck -check-prefix=CHECK-NANLEGACY %s
// RUN: FileCheck -check-prefix=CHECK-MIPS4 %s < %t
//
// RUN: %clang -target mipsel-unknown-linux -mnan=2008 -march=mips32 -emit-llvm -S %s -o - 2>%t | FileCheck -check-prefix=CHECK-NANLEGACY %s
// RUN: FileCheck -check-prefix=CHECK-MIPS32 %s < %t
//
// RUN: %clang -target mipsel-unknown-linux -mnan=2008 -march=mips32r2 -emit-llvm -S %s -o - 2>%t | FileCheck -check-prefix=CHECK-NAN2008 %s
// RUN: FileCheck -allow-empty -check-prefix=NO-WARNINGS %s < %t
//
// RUN: %clang -target mipsel-unknown-linux -mnan=2008 -march=mips32r3 -emit-llvm -S %s -o - 2>%t | FileCheck -check-prefix=CHECK-NAN2008 %s
// RUN: FileCheck -allow-empty -check-prefix=NO-WARNINGS %s < %t
//
// RUN: %clang -target mipsel-unknown-linux -mnan=legacy -march=mips32r6 -emit-llvm -S %s -o - 2>%t | FileCheck -check-prefix=CHECK-NAN2008 %s
// RUN: FileCheck -check-prefix=CHECK-MIPS32R6 %s < %t
//
// RUN: %clang -target mips64el-unknown-linux -mnan=2008 -march=mips64 -emit-llvm -S %s -o - 2>%t | FileCheck -check-prefix=CHECK-NANLEGACY %s
// RUN: FileCheck -check-prefix=CHECK-MIPS64 %s < %t
//
// RUN: %clang -target mips64el-unknown-linux -mnan=2008 -march=mips64r2 -emit-llvm -S %s -o - 2>%t | FileCheck -check-prefix=CHECK-NAN2008 %s
// RUN: FileCheck -allow-empty -check-prefix=NO-WARNINGS %s < %t
//
// RUN: %clang -target mips64el-unknown-linux -mnan=legacy -march=mips64r6 -emit-llvm -S %s -o - 2>%t | FileCheck -check-prefix=CHECK-NAN2008 %s
// RUN: FileCheck -check-prefix=CHECK-MIPS64R6 %s < %t

// NO-WARNINGS-NOT: warning: ignoring '-mnan=legacy' option
// NO-WARNINGS-NOT: warning: ignoring '-mnan=2008' option

// CHECK-MIPS2: warning: ignoring '-mnan=2008' option because the 'mips2' architecture does not support it
// CHECK-MIPS3: warning: ignoring '-mnan=2008' option because the 'mips3' architecture does not support it
// CHECK-MIPS4: warning: ignoring '-mnan=2008' option because the 'mips4' architecture does not support it
// CHECK-MIPS32: warning: ignoring '-mnan=2008' option because the 'mips32' architecture does not support it
// CHECK-MIPS32R6: warning: ignoring '-mnan=legacy' option because the 'mips32r6' architecture does not support it
// CHECK-MIPS64: warning: ignoring '-mnan=2008' option because the 'mips64' architecture does not support it
// CHECK-MIPS64R6: warning: ignoring '-mnan=legacy' option because the 'mips64r6' architecture does not support it

// This call creates a QNAN double with an empty payload.
// The quiet bit is inverted in legacy mode: it is clear to indicate QNAN,
// so the next highest bit is set to maintain NAN (not infinity).
// In regular (2008) mode, the quiet bit is set to indicate QNAN.

// CHECK-NANLEGACY: double 0x7FF4000000000000
// CHECK-NAN2008: double 0x7FF8000000000000

double d =  __builtin_nan("");

// This call creates a QNAN double with an empty payload and then truncates.
// llvm::APFloat does not know about the inverted quiet bit, so it sets the
// quiet bit on conversion independently of the setting in clang.

// CHECK-NANLEGACY: float 0x7FFC000000000000
// CHECK-NAN2008: float 0x7FF8000000000000

float f =  __builtin_nan("");
