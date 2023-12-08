// ================== Check default Architecture on krait CPU
// RUN: %clang -target arm-linux-gnueabi -mcpu=krait -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CPUV7A %s
// ================== Check whether -mcpu accepts mixed-case values.
// RUN: %clang -target arm-linux-gnueabi -mcpu=Krait -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CPUV7A %s
// RUN: %clang -target arm-linux-gnueabi -mcpu=KRAIT -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CPUV7A %s
// CHECK-CPUV7A: "-cc1"{{.*}} "-triple" "armv7-{{.*}}
