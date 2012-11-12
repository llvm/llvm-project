// Check passing options to the assembler for ARM targets.
//
// RUN: %clang -target arm-linux -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM %s
// CHECK-ARM: as{{(.exe)?}}" "-mfloat-abi=soft"
//
// RUN: %clang -target arm-linux -mcpu=cortex-a8 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-MCPU %s
// CHECK-ARM-MCPU: as{{(.exe)?}}" "-mfloat-abi=soft" "-mcpu=cortex-a8"
//
// RUN: %clang -target arm-linux -mfpu=neon -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-MFPU %s
// CHECK-ARM-MFPU: as{{(.exe)?}}" "-mfloat-abi=soft" "-mfpu=neon"
//
// RUN: %clang -target arm-linux -march=armv7-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-MARCH %s
// CHECK-ARM-MARCH: as{{(.exe)?}}" "-mfloat-abi=soft" "-march=armv7-a"
//
// RUN: %clang -target arm-linux -mcpu=cortex-a8 -mfpu=neon -march=armv7-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-ALL %s
// CHECK-ARM-ALL: as{{(.exe)?}}" "-mfloat-abi=soft" "-march=armv7-a" "-mcpu=cortex-a8" "-mfpu=neon"
//
// RUN: %clang -target armv7-linux -mcpu=cortex-a8 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-TARGET %s
// CHECK-ARM-TARGET: as{{(.exe)?}}" "-mfpu=neon" "-mfloat-abi=soft" "-mcpu=cortex-a8"
//
// RUN: %clang -target arm-linux -mfloat-abi=hard -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-MFLOAT-ABI %s
// CHECK-ARM-MFLOAT-ABI: as{{(.exe)?}}" "-mfloat-abi=hard"
//
// RUN: %clang -target arm-linux-androideabi -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-ANDROID %s
// CHECK-ARM-ANDROID: as{{(.exe)?}}" "-mfloat-abi=soft"
//
// RUN: %clang -target arm-linux-androideabi -march=armv7-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-ANDROID-SOFTFP %s
// CHECK-ARM-ANDROID-SOFTFP: as{{(.exe)?}}" "-mfloat-abi=softfp" "-march=armv7-a"
//
// RUN: %clang -target arm-linux-eabi -mhard-float -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-HARDFP %s
// CHECK-ARM-HARDFP: as{{(.exe)?}}" "-mfloat-abi=hard"
//
// RUN: %clang -target ppc-linux -mcpu=invalid-cpu -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=PPC-NO-MCPU %s
// CHECK-PPC-NO-MCPU-NOT: as{{.*}} "-mcpu=invalid-cpu"
