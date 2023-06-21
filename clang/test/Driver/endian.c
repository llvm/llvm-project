// RUN: %clang -### -c --target=x86_64 -mbig-endian -mlittle-endian %s 2>&1 | FileCheck /dev/null --implicit-check-not=error:
// RUN: %clang -### -c --target=x86_64 -mlittle-endian -mbig-endian %s 2>&1 | FileCheck %s --implicit-check-not=error:

// CHECK: error: unsupported option '-mlittle-endian' for target 'x86_64'
// CHECK: error: unsupported option '-mbig-endian' for target 'x86_64'
