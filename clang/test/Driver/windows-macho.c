// RUN: %clang -target armv7-pc-win32-macho -msoft-float -### -c %s 2>&1 \
// RUN: | FileCheck %s --check-prefix CHECK-SOFTFLOAT
// CHECK-SOFTFLOAT-NOT: error: unsupported option '-msoft-float' for target 'thumbv7-pc-windows-macho'

// RUN: not %clang --target=armv7-pc-win32-macho -mhard-float -### -c %s 2>&1 \
// RUN: | FileCheck %s --check-prefix CHECK-HARDFLOAT
// CHECK-HARDFLOAT: error: unsupported option '-mhard-float' for target 'thumbv7-pc-windows-macho'

// RUN: %clang -target armv7-pc-win32-macho -### -c %s 2>&1 \
// RUN: | FileCheck %s --check-prefix CHECK-DEFAULT-SOFTFLOAT-ABI
// CHECK-DEFAULT-SOFTFLOAT-ABI: "-mfloat-abi" "soft"
