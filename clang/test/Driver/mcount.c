// RUN: %clang -target s390x -c -### %s -mnop-mcount -mrecord-mcount 2>&1 | FileCheck %s

// CHECK: "-mnop-mcount"
// CHECK: "-mrecord-mcount"

// RUN: %clang -target x86_64 -c -### %s -mnop-mcount -mrecord-mcount 2>&1 | FileCheck --check-prefix=ERR1 %s
// RUN: %clang -target aarch64 -c -### %s -mnop-mcount -mrecord-mcount 2>&1 | FileCheck --check-prefix=ERR2 %s

// ERR1: error: unsupported option '-mnop-mcount' for target 'x86_64'
// ERR1: error: unsupported option '-mrecord-mcount' for target 'x86_64'
// ERR2: error: unsupported option '-mnop-mcount' for target 'aarch64'
// ERR2: error: unsupported option '-mrecord-mcount' for target 'aarch64'
