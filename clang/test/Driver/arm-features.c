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

// Check that crypto means sha2 + aes:
//
// Check +sha2 +aes:
//
// RUN: %clang -target arm-arm-none-eabi -march=armv8.2a+sha2+aes -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-SHA2-AES %s
// CHECK-SHA2-AES: "-cc1"{{.*}} "-triple" "armv8.2a-{{.*}} "-target-cpu" "generic"{{.*}} "-target-feature" "+sha2" "-target-feature" "+aes"
//
// Check -sha2 -aes:
//
// RUN: %clang -target arm-arm-none-eabi -march=armv8.2a+sha2+nosha2+aes+noaes -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SHA2-AES %s
// CHECK-NO-SHA2-AES: "-cc1"{{.*}} "-triple" "armv8.2a-{{.*}} "-target-cpu" "generic"{{.*}} "-target-feature" "-sha2" "-target-feature" "-aes"
//
// Check +crypto:
//
// RUN: %clang -target arm-arm-none-eabi -march=armv8a+crypto   -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO2 %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1a+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO2 %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8.2a+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO2 %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8.3a+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO2 %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8.4a+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO2 %s
// CHECK-CRYPTO2: "-cc1"{{.*}} "-target-cpu" "generic"{{.*}} "-target-feature" "+crypto" "-target-feature" "+sha2" "-target-feature" "+aes"
//
// Check -crypto:
//
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO2 %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8.2a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO2 %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8.3a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO2 %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8.4a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO2 %s
// CHECK-NOCRYPTO2-NOT: "-target-feature" "+crypto" "-target-feature" "+sha2" "-target-feature" "+aes"
//
// Check +crypto -sha2 -aes:
//
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1a+crypto+nosha2+noaes -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO3 %s
// CHECK-CRYPTO3-NOT: "-target-feature" "+sha2" "-target-feature" "+aes"
//
// Check -crypto +sha2 +aes:
//
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1a+nocrypto+sha2+aes -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO4 %s
// CHECK-CRYPTO4: "-target-feature" "+sha2" "-target-feature" "+aes"
