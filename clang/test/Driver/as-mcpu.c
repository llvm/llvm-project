// ================== Check that krait is substituted by cortex-a15 when invoking
// the assembler
// RUN: %clang -target arm-linux -mcpu=krait -### -c %s -v -fno-integrated-as 2>&1 | FileCheck -check-prefix=CHECK-CORTEX-A15 %s
// CHECK-CORTEX-A15: as{{(.exe)?}}" "{{.*}}-mcpu=cortex-a15

// ================== Check that kryo is substituted by cortex-a57 when invoking
// the assembler
// RUN: %clang -target arm-linux -mcpu=kryo -### -c %s -v -fno-integrated-as 2>&1 | FileCheck -check-prefix=CHECK-CORTEX-A57 %s
// RUN: %clang -target aarch64-linux -mcpu=kryo -### -c %s -v -fno-integrated-as 2>&1 | FileCheck -check-prefix=CHECK-CORTEX-A57 %s

// CHECK-CORTEX-A57: as{{(.exe)?}}" "{{.*}}-mcpu=cortex-a57
