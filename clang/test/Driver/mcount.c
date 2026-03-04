// RUN: %clang --target=s390x -c -### %s -mnop-mcount -mrecord-mcount 2>&1 | FileCheck %s
// RUN: %clang --target=x86_64 -c -### %s -mnop-mcount -mrecord-mcount 2>&1 | FileCheck %s

// CHECK: "-mnop-mcount"
// CHECK: "-mrecord-mcount"

// RUN: not %clang --target=aarch64 -c -### %s -mnop-mcount -mrecord-mcount 2>&1 | FileCheck --check-prefix=ERR2 %s

// ERR2: error: unsupported option '-mnop-mcount' for target 'aarch64'
// ERR2: error: unsupported option '-mrecord-mcount' for target 'aarch64'
