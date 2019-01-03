// RUN: clang-check "%s" -extra-arg=-Wunimplemented-warning -extra-arg-before=-Wunimplemented-warning-before -- -c 2>&1 | FileCheck %s

// CHECK: unknown warning option '-Wunimplemented-warning-before'
// CHECK: unknown warning option '-Wunimplemented-warning'

// Check we do not crash with -extra-arg=-gsplit-dwarf (we did, under linux).
// RUN: clang-check "%s" -extra-arg=-gsplit-dwarf -- -c

void a(){}
