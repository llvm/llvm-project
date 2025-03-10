// RUN: not %clang --target=x86_64 -c -mcmodel=tiny %s 2>&1 | FileCheck %s
// CHECK: error: unsupported argument 'tiny' to option '-mcmodel=' for target '{{.*}}'
