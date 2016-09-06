// RUN: %clang -target arm-none-none-eabi -mcpu=generic+crc -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRC %s
// RUN: %clang -target arm-none-none-eabi -mcpu=generic -march=armv8a+crc -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRC %s
// CHECK-CRC: "-cc1"{{.*}} "-triple" "armv8-{{.*}} "-target-cpu" "generic"{{.*}} "-target-feature" "+crc"
// RUN: %clang -target arm-none-none-eabi -mcpu=generic+crypto -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO %s
// RUN: %clang -target arm-none-none-eabi -mcpu=generic -march=armv8a+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO %s
// CHECK-CRYPTO: "-cc1"{{.*}} "-triple" "armv8-{{.*}} "-target-cpu" "generic"{{.*}} "-target-feature" "+crypto"
// RUN: %clang -target arm-none-none-eabi -mcpu=generic+dsp -march=armv8m.main -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-DSP %s
// RUN: %clang -target arm-none-none-eabi -mcpu=generic -march=armv8m.main+dsp -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-DSP %s
// CHECK-DSP: "-cc1"{{.*}} "-triple" "thumbv8m.main-{{.*}} "-target-cpu" "generic"{{.*}} "-target-feature" "+dsp"

// RUN: %clang -target arm-none-none-eabi -mcpu=generic+nocrc -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRC %s
// RUN: %clang -target arm-none-none-eabi -mcpu=generic -march=armv8a+nocrc -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRC %s
// CHECK-NOCRC: "-cc1"{{.*}} "-triple" "armv8-{{.*}} "-target-cpu" "generic"{{.*}} "-target-feature" "-crc"
// RUN: %clang -target arm-none-none-eabi -mcpu=generic+nocrypto -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO %s
// RUN: %clang -target arm-none-none-eabi -mcpu=generic -march=armv8a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO %s
// CHECK-NOCRYPTO: "-cc1"{{.*}} "-triple" "armv8-{{.*}} "-target-cpu" "generic"{{.*}} "-target-feature" "-crypto"
// RUN: %clang -target arm-none-none-eabi -mcpu=generic+nodsp -march=armv8m.main -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NODSP %s
// RUN: %clang -target arm-none-none-eabi -mcpu=generic -march=armv8m.main+nodsp -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NODSP %s
// CHECK-NODSP: "-cc1"{{.*}} "-triple" "thumbv8m.main-{{.*}} "-target-cpu" "generic"{{.*}} "-target-feature" "-dsp"
