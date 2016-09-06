// Test that the timestamp is not included in the produced pch file with
// -fno-pch-timestamp.

// Copying files allow for read-only checkouts to run this test.
// RUN: cp %S/Inputs/pragma-once2-pch.h %T
// RUN: cp %S/Inputs/pragma-once2.h %T
// RUN: cp %s %t1.cpp

// Check timestamp is included by default.
// RUN: %clang_cc1 -x c++-header -emit-pch -o %t %T/pragma-once2-pch.h
// RUN: touch -m -a -t 201008011501 %T/pragma-once2.h
// RUN: not %clang_cc1 -include-pch %t %t1.cpp 2>&1 | FileCheck -check-prefix=CHECK-TIMESTAMP %s

// Check bitcode output as well.
// RUN: llvm-bcanalyzer -dump %t | FileCheck -check-prefix=CHECK-BITCODE-TIMESTAMP-ON %s

// Check timestamp inclusion is disabled by -fno-pch-timestamp.
// RUN: %clang_cc1 -x c++-header -emit-pch -o %t %T/pragma-once2-pch.h -fno-pch-timestamp
// RUN: touch -m -a -t 201008011502 %T/pragma-once2.h
// RUN: %clang_cc1 -include-pch %t %t1.cpp 2>&1

// Check bitcode output as well.
// RUN: llvm-bcanalyzer -dump %t | FileCheck -check-prefix=CHECK-BITCODE-TIMESTAMP-OFF %s

#include "pragma-once2.h"

void g() { f(); }

// CHECK-BITCODE-TIMESTAMP-ON: <INPUT_FILE abbrevid={{.*}} op0={{.*}} op1={{.*}} op2={{[^0]}}
// CHECK-BITCODE-TIMESTAMP-OFF: <INPUT_FILE abbrevid={{.*}} op0={{.*}} op1={{.*}} op2={{[0]}}

// CHECK-TIMESTAMP: fatal error: file {{.*}} has been modified since the precompiled header {{.*}} was built
