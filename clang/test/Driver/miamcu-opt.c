// RUN: %clang -### --target=x86_64 -miamcu -rtlib=platform --unwindlib=platform %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=x86_64 -miamcu -rtlib=platform --unwindlib=platform -m32 %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=x86_64 -miamcu -rtlib=platform --unwindlib=platform --target=x86_64-unknown-linux-gnu %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=x86_64 -mno-iamcu -miamcu -rtlib=platform --unwindlib=platform %s 2>&1 | FileCheck %s
// RUN: not %clang -### --target=x86_64 -miamcu -rtlib=platform --unwindlib=platform -m64 %s 2>&1 | FileCheck %s -check-prefix=M64
// RUN: not %clang -### --target=x86_64 -miamcu -rtlib=platform --unwindlib=platform -dynamic %s 2>&1 | FileCheck %s -check-prefix=DYNAMIC
// RUN: not %clang -### --target=x86_64 -miamcu -rtlib=platform --unwindlib=platform  --target=armv8-eabi %s 2>&1 | FileCheck %s -check-prefix=NOT-X86
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -miamcu -mno-iamcu %s 2>&1 | FileCheck %s -check-prefix=MNOIAMCU

// M64: error: invalid argument '-miamcu' not allowed with '-m64'

// DYNAMIC: error: invalid argument '-dynamic' not allowed with '-static'

// NOT-X86: error: unsupported option '-miamcu' for target 'armv8-unknown-unknown-eabi'

// MNOIAMCU-NOT: "-triple" "i586-intel-elfiamcu"

// CHECK: "-cc1"
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
