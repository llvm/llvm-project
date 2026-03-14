// Test user-facing command line options to generate C++20 header units.

// RUN: %clang -### -std=c++20 -fmodule-header=user foo.hh  2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-USER %s

// RUN: %clang -### -std=c++20 -fmodule-header=user foo.h  2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-USER1 %s

// RUN: %clang -### -std=c++20 -fmodule-header=system foo.hh 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-SYS1 %s

// RUN: %clang -### -std=c++20 -fmodule-header=system \
// RUN: -xc++-system-header vector 2>&1 | FileCheck -check-prefix=CHECK-SYS2 %s

// RUN: %clang -### -std=c++20 -fmodule-header=system \
// RUN: -xc++-header vector 2>&1 | FileCheck -check-prefix=CHECK-SYS2 %s

// RUN: %clang -### -std=c++20 -fmodule-header %/S/Inputs/header-unit-01.hh \
// RUN: 2>&1 | FileCheck -check-prefix=CHECK-ABS %s -DTDIR=%/S/Inputs

// CHECK-USER: "-emit-header-unit"
// CHECK-USER-SAME: "-o" "foo.pcm"
// CHECK-USER-SAME: "-x" "c++-user-header" "foo.hh"

// CHECK-USER1: "-emit-header-unit"
// CHECK-USER1-SAME: "-o" "foo.pcm"
// CHECK-USER1-SAME: "-x" "c++-user-header" "foo.h"

// CHECK-SYS1: "-emit-header-unit"
// CHECK-SYS1-SAME: "-o" "foo.pcm"
// CHECK-SYS1-SAME: "-x" "c++-system-header" "foo.hh"

// CHECK-SYS2: "-emit-header-unit"
// CHECK-SYS2-SAME: "-o" "vector.pcm"
// CHECK-SYS2-SAME: "-x" "c++-system-header" "vector"

// CHECK-ABS: "-emit-header-unit"
// CHECK-ABS-SAME: "-o" "header-unit-01.pcm"
// CHECK-ABS-SAME: "-x" "c++-header-unit-header" "[[TDIR]]/header-unit-01.hh"
