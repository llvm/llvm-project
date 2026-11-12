// RUN: %clang -### -std=c++20 -xc++-user-header foo.h 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-USER %s

// RUN: %clang -### -std=c++20 -xc++-system-header vector 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-SYSTEM %s

// RUN: %clang -### -std=c++20 \
// RUN:   -xc++-header-unit-header %/S/Inputs/header-unit-01.hh 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-ABS %s -DTDIR=%/S/Inputs

// CHECK-USER: "-emit-header-unit"
// CHECK-USER-SAME: "-o" "foo.pcm"
// CHECK-USER-SAME: "-x" "c++-user-header" "foo.h"
// CHECK-SYSTEM: "-emit-header-unit"
// CHECK-SYSTEM-SAME: "-o" "vector.pcm"
// CHECK-SYSTEM-SAME: "-x" "c++-system-header" "vector"
// CHECK-ABS: "-emit-header-unit"
// CHECK-ABS-SAME: "-o" "header-unit-01.pcm"
// CHECK-ABS-SAME: "-x" "c++-header-unit-header" "[[TDIR]]/header-unit-01.hh"
