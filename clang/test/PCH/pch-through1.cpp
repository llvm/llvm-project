// Through header not found (anywhere)
// RUN: not %clang_cc1 -emit-pch \
// RUN:   -pch-through-header=Inputs/pch-does-not-exist.h -o %t %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-TEST0A %s
// CHECK-TEST0A: fatal error:{{.*}} 'Inputs/pch-does-not-exist.h'
// CHECK-TEST0A-SAME: required for precompiled header not found

// Through header not found in search path
// RUN: not %clang_cc1 -emit-pch \
// RUN:   -pch-through-header=Inputs/pch-through2.h -o %t \
// RUN:   %S/Inputs/pch-through-use0.cpp 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-TEST0B %s
// CHECK-TEST0B: fatal error:{{.*}}'Inputs/pch-through2.h'
// CHECK-TEST0B-SAME: required for precompiled header not found

// No #include of through header during pch create
// RUN: not %clang_cc1 -I %S -emit-pch \
// RUN:   -pch-through-header=Inputs/pch-through2.h -o %t %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-TEST1A %s
// CHECK-TEST1A: fatal error:{{.*}} #include of
// CHECK-TEST1A-SAME: 'Inputs/pch-through2.h' not seen while attempting to
// CHECK-TEST1A-SAME: create precompiled header

// checks for through headers that are also -includes
// RUN: %clang_cc1 -I %S -include Inputs/pch-through1.h \
// RUN:   -pch-through-header=Inputs/pch-through1.h -emit-pch -o %t.s3t1 %s
// RUN: %clang_cc1 -I %S -include Inputs/pch-through1.h \
// RUN:   -include Inputs/pch-through2.h -include Inputs/pch-through3.h \
// RUN:   -pch-through-header=Inputs/pch-through2.h -emit-pch -o %t.s3t2 %s
// Use through header from -includes
// RUN: %clang_cc1 -I %S -include Inputs/pch-through1.h \
// RUN:   -include Inputs/pch-through2.h -include Inputs/pch-through4.h \
// RUN:   -pch-through-header=Inputs/pch-through2.h -include-pch %t.s3t2 \
// RUN:   %S/Inputs/pch-through-use2.cpp -o %t.out
