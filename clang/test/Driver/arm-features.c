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
// RUN: %clang -target arm-arm-none-eabi -march=armv8.5a+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO2 %s
// CHECK-CRYPTO2: "-cc1"{{.*}} "-target-cpu" "generic"{{.*}} "-target-feature" "+crypto"{{.*}} "-target-feature" "+sha2" "-target-feature" "+aes"
//
// Check -crypto:
//
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO2 %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8.2a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO2 %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8.3a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO2 %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8.4a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO2 %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8.5a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO2 %s
// CHECK-NOCRYPTO2-NOT: "-target-feature" "+crypto" "-target-feature" "+sha2" "-target-feature" "+aes"
//
// RUN: %clang -target arm-arm-none-eabi -mcpu=cortex-a57+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO2-CPU %s
// RUN: %clang -target arm-arm-none-eabi -mcpu=cortex-a57 -mfpu=crypto-neon-fp-armv8 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO2-CPU %s
// CHECK-CRYPTO2-CPU: "-cc1"{{.*}} "-target-cpu" "cortex-a57"{{.*}} "-target-feature" "+crypto"{{.*}} "-target-feature" "+sha2" "-target-feature" "+aes"
//
// RUN: %clang -target arm-arm-none-eabi -mcpu=cortex-a57+norypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO2-CPU %s
// RUN: %clang -target arm-arm-none-eabi -mcpu=cortex-a57 -mfpu=neon-fp-armv8 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO2-CPU %s
// CHECK-NOCRYPTO2-CPU-NOT: "-cc1"{{.*}} "-target-cpu" "cortex-a57"{{.*}} "-target-feature" "+crypto"{{.*}} "-target-feature" "+sha2" "-target-feature" "+aes"
//
// Check +crypto -sha2 -aes:
//
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1a+crypto+nosha2+noaes -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO3 %s
// RUN: %clang -target arm-arm-none-eabi -mcpu=cortex-a57+crypto+nosha2+noaes -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO3 %s
// RUN: %clang -target arm-arm-none-eabi -mcpu=cortex-a57+nosha2+noaes -mfpu=crypto-neon-fp-armv8 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO3 %s
// CHECK-CRYPTO3-NOT: "-target-feature" "+sha2" "-target-feature" "+aes"
//
// Check -crypto +sha2 +aes:
//
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1a+nocrypto+sha2+aes -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO4 %s
// RUN: %clang -target arm-arm-none-eabi -mcpu=cortex-a57+nocrypto+sha2+aes -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO4 %s
// RUN: %clang -target arm-arm-none-eabi -mcpu=cortex-a57+sha2+aes -mfpu=neon-fp-armv8 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO4 %s
// CHECK-CRYPTO4: "-target-feature" "+sha2" "-target-feature" "+aes"
//
// Check +crypto for M and R profiles:
//
// RUN: %clang -target arm-arm-none-eabi -march=armv8-r+crypto   -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO5 %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8-m.base+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO5 %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8-m.main+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO5 %s
// RUN: %clang -target arm-arm-none-eabi -mcpu=cortex-m23+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO5 %s
// CHECK-NOCRYPTO5: warning: ignoring extension 'crypto' because the {{.*}} architecture does not support it
// CHECK-NOCRYPTO5-NOT: "-target-feature" "+crypto"{{.*}} "-target-feature" "+sha2" "-target-feature" "+aes"
//
// Check +crypto does not affect -march=armv7a -mfpu=crypto-neon-fp-armv8, but it does warn that +crypto has no effect
// RUN: %clang -target arm-none-none-eabi -fno-integrated-as -march=armv7a -mfpu=crypto-neon-fp-armv8 -### -c %s 2>&1 | FileCheck -check-prefixes=CHECK-WARNONLY,ALL %s
// RUN: %clang -target arm-none-none-eabi -fno-integrated-as -march=armv7a+aes -mfpu=crypto-neon-fp-armv8 -### -c %s 2>&1 | FileCheck -check-prefixes=CHECK-WARNONLY,ALL,CHECK-HASAES %s
// RUN: %clang -target arm-none-none-eabi -fno-integrated-as -march=armv7a+sha2 -mfpu=crypto-neon-fp-armv8 -### -c %s 2>&1 | FileCheck -check-prefixes=CHECK-WARNONLY,ALL,CHECK-HASSHA %s
// RUN: %clang -target arm-none-none-eabi -fno-integrated-as -march=armv7a+sha2+aes -mfpu=crypto-neon-fp-armv8 -### -c %s 2>&1 | FileCheck -check-prefixes=CHECK-WARNONLY,ALL,CHECK-HASSHA,CHECK-HASAES %s
// RUN: %clang -target arm-none-none-eabi -fno-integrated-as -march=armv7a+aes -mfpu=neon-fp-armv8 -### -c %s 2>&1 | FileCheck -check-prefixes=ALL,CHECK-HASAES %s
// RUN: %clang -target arm-none-none-eabi -fno-integrated-as -march=armv7a+sha2 -mfpu=neon-fp-armv8 -### -c %s 2>&1 | FileCheck -check-prefixes=ALL,CHECK-HASSHA %s
// RUN: %clang -target arm-none-none-eabi -fno-integrated-as -march=armv7a+sha2+aes -mfpu=neon-fp-armv8 -### -c %s 2>&1 | FileCheck -check-prefixes=ALL,CHECK-HASSHA,CHECK-HASAES %s
// CHECK-WARNONLY: warning: ignoring extension 'crypto' because the 'armv7-a' architecture does not support it
// ALL:     "-target-feature"
// CHECK-WARNONLY-NOT: "-target-feature" "-crypto"
// CHECK-HASSHA-SAME:  "-target-feature" "+sha2"
// CHECK-HASAES-SAME:  "-target-feature" "+aes"
//
