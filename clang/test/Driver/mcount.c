// RUN: %clang -target s390x -c -### %s -mrecord-mcount 2>&1 | FileCheck %s
// CHECK: "-mrecord-mcount"

// RUN: %clang -target x86_64 -c -### %s -mrecord-mcount 2>&1 | FileCheck --check-prefix=ERR1 %s
// RUN: %clang -target aarch64 -c -### %s -mrecord-mcount 2>&1 | FileCheck --check-prefix=ERR2 %s

// ERR1: error: unsupported option '-mrecord-mcount' for target 'x86_64'
// ERR2: error: unsupported option '-mrecord-mcount' for target 'aarch64'
