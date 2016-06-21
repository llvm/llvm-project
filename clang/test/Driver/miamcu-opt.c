// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
//
// RUN: %clang -miamcu -no-canonical-prefixes %s -### -o %t.o 2>&1 | FileCheck %s
// RUN: %clang -miamcu -no-canonical-prefixes -m32 %s -### -o %t.o 2>&1 | FileCheck %s
// RUN: %clang -miamcu -no-canonical-prefixes -target x86_64-unknown-linux-gnu %s -### -o %t.o 2>&1 | FileCheck %s
// RUN: %clang -mno-iamcu -miamcu -no-canonical-prefixes %s -### -o %t.o 2>&1 | FileCheck %s
// RUN: %clang -miamcu -no-canonical-prefixes -m64 %s -### -o %t.o 2>&1 | FileCheck %s -check-prefix=M64
// RUN: %clang -miamcu -no-canonical-prefixes -dynamic %s -### -o %t.o 2>&1 | FileCheck %s -check-prefix=DYNAMIC
// RUN: %clang -miamcu -no-canonical-prefixes  -target armv8-eabi %s -### -o %t.o 2>&1 | FileCheck %s -check-prefix=NOT-X86
// RUN: %clang -miamcu -mno-iamcu -no-canonical-prefixes -target x86_64-unknown-linux-gnu %s -### -o %t.o 2>&1 | FileCheck %s -check-prefix=MNOIAMCU

// M64: error: invalid argument '-miamcu' not allowed with '-m64'

// DYNAMIC: error: invalid argument '-dynamic' not allowed with '-static'

// NOT-X86: error: unsupported option '-miamcu' for target 'armv8---eabi'

// MNOIAMCU-NOT: "-triple" "i586-intel-elfiamcu"

// CHECK: "{{.*}}clang{{.*}}" "-cc1"
// CHECK: "-triple" "i586-intel-elfiamcu"
// CHECK: "-static-define"
// CHECK: "-mfloat-abi" "soft"
// CHECK: "-mstack-alignment=4"

// CHECK: "{{.*}}ld{{(.exe)?}}"
// CHECK: "-m" "elf_iamcu"
// CHECK: "-static"
// CHECK-NOT: crt1
// CHECK-NOT: crti
// CHECK-NOT: ctrbegin
// CHECK: crt0
// CHECK: "--start-group" "-lgcc" "-lc" "-lgloss" "--end-group" "--as-needed" "-lsoftfp" "--no-as-needed"
// CHECK-NOT: crtend
// CHECK-NOT: ctrn
