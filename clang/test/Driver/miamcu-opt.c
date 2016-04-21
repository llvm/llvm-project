// RUN: %clang -miamcu %s -### -o %t.o 2>&1 | FileCheck %s
// RUN: %clang -miamcu -m32 %s -### -o %t.o 2>&1 | FileCheck %s
// RUN: %clang -miamcu -target x86_64-unknown-linux-gnu %s -### -o %t.o 2>&1 | FileCheck %s
// RUN: %clang -miamcu -m64 %s -### -o %t.o 2>&1 | FileCheck %s -check-prefix=M64
// RUN: %clang -miamcu -dynamic %s -### -o %t.o 2>&1 | FileCheck %s -check-prefix=DYNAMIC
// RUN: %clang -miamcu -target armv8-eabi %s -### -o %t.o 2>&1 | FileCheck %s -check-prefix=NOT-X86

// M64: error: invalid argument '-miamcu' not allowed with '-m64'

// DYNAMIC: error: invalid argument '-dynamic' not allowed with '-static'

// NOT-X86: error: unsupported option '-miamcu' for target 'armv8---eabi'

// CHECK: "-cc1"
// CHECK: "-triple" "i586-intel-elfiamcu"
// CHECK: "-static-define"
// CHECK: "-mfloat-abi" "soft"
// CHECK: "-mstack-alignment=4"
