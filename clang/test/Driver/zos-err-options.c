// RUN: not %clang -msoft-float --target=s390x-none-zos 2>&1 %s | FileCheck %s

// CHECK: error: unsupported option '-msoft-float' for target 's390x-none-zos'
