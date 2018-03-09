// Check passing options to the assembler for various linux targets.
//
// RUN: %clang -target arm-linux -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM %s
// CHECK-ARM: as{{(.exe)?}}" "-mfloat-abi=soft"
//
// RUN: %clang -target arm-linux -mcpu=cortex-a8 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-MCPU %s
// CHECK-ARM-MCPU: as{{(.exe)?}}" "-mfloat-abi=soft" "-mcpu=cortex-a8"
//
// RUN: %clang -target arm-linux -mfpu=neon -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-MFPU %s
// CHECK-ARM-MFPU: as{{(.exe)?}}" "-mfloat-abi=soft" "-mfpu=neon"
//
// RUN: %clang -target arm-linux -march=armv7-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-MARCH %s
// CHECK-ARM-MARCH: as{{(.exe)?}}" "-mfloat-abi=soft" "-march=armv7-a"
//
// RUN: %clang -target arm-linux -mcpu=cortex-a8 -mfpu=neon -march=armv7-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-ALL %s
// CHECK-ARM-ALL: as{{(.exe)?}}" "-mfloat-abi=soft" "-march=armv7-a" "-mcpu=cortex-a8" "-mfpu=neon"
//
// RUN: %clang -target arm-linux -mcpu=cortex-a8 -mfpu=neon -march=armebv7-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARMEB-ALL %s
// CHECK-ARMEB-ALL: as{{(.exe)?}}" "-mfloat-abi=soft" "-march=armebv7-a" "-mcpu=cortex-a8" "-mfpu=neon"
//
// RUN: %clang -target thumb-linux -mcpu=cortex-a8 -mfpu=neon -march=thumbv7-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-THUMB-ALL %s
// CHECK-THUMB-ALL: as{{(.exe)?}}" "-mfloat-abi=soft" "-march=thumbv7-a" "-mcpu=cortex-a8" "-mfpu=neon"
//
// RUN: %clang -target thumb-linux -mcpu=cortex-a8 -mfpu=neon -march=thumbebv7-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-THUMBEB-ALL %s
// CHECK-THUMBEB-ALL: as{{(.exe)?}}" "-mfloat-abi=soft" "-march=thumbebv7-a" "-mcpu=cortex-a8" "-mfpu=neon"
//
// RUN: %clang -target armv7-linux -mcpu=cortex-a8 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-TARGET %s
// CHECK-ARM-TARGET: as{{(.exe)?}}" "-mfpu=neon" "-mfloat-abi=soft" "-mcpu=cortex-a8"
//
// RUN: %clang -target armebv7-linux -mcpu=cortex-a8 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARMEB-TARGET %s
// CHECK-ARMEB-TARGET: as{{(.exe)?}}" "-mfpu=neon" "-mfloat-abi=soft" "-mcpu=cortex-a8"
//
// RUN: %clang -target thumbv7-linux -mcpu=cortex-a8 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-THUMB-TARGET %s
// CHECK-THUMB-TARGET: as{{(.exe)?}}" "-mfpu=neon" "-mfloat-abi=soft" "-mcpu=cortex-a8"
//
// RUN: %clang -target thumbebv7-linux -mcpu=cortex-a8 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-THUMBEB-TARGET %s
// CHECK-THUMBEB-TARGET: as{{(.exe)?}}" "-mfpu=neon" "-mfloat-abi=soft" "-mcpu=cortex-a8"
//
// RUN: %clang -target armv8-linux -mcpu=cortex-a53 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-TARGET-V8 %s
// CHECK-ARM-TARGET-V8: as{{(.exe)?}}" "-mfpu=crypto-neon-fp-armv8" "-mfloat-abi=soft" "-mcpu=cortex-a53"
//
// RUN: %clang -target armebv8-linux -mcpu=cortex-a53 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARMEB-TARGET-V8 %s
// CHECK-ARMEB-TARGET-V8: as{{(.exe)?}}" "-mfpu=crypto-neon-fp-armv8" "-mfloat-abi=soft" "-mcpu=cortex-a53"
//
// RUN: %clang -target thumbv8-linux -mcpu=cortex-a53 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-THUMB-TARGET-V8 %s
// CHECK-THUMB-TARGET-V8: as{{(.exe)?}}" "-mfpu=crypto-neon-fp-armv8" "-mfloat-abi=soft" "-mcpu=cortex-a53"
//
// RUN: %clang -target thumbebv8-linux -mcpu=cortex-a53 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-THUMBEB-TARGET-V8 %s
// CHECK-THUMBEB-TARGET-V8: as{{(.exe)?}}" "-mfpu=crypto-neon-fp-armv8" "-mfloat-abi=soft" "-mcpu=cortex-a53"
//
// RUN: %clang -target arm-linux -mfloat-abi=hard -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-MFLOAT-ABI %s
// CHECK-ARM-MFLOAT-ABI: as{{(.exe)?}}" "-mfloat-abi=hard"
//
// RUN: %clang -target arm-linux-androideabi -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-ANDROID %s
// CHECK-ARM-ANDROID: as{{(.exe)?}}" "-mfloat-abi=soft"
//
// RUN: %clang -target arm-linux-androideabi -march=armv7-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-ANDROID-SOFTFP %s
// CHECK-ARM-ANDROID-SOFTFP: as{{(.exe)?}}" "-mfloat-abi=softfp" "-march=armv7-a"
//
// RUN: %clang -target arm-linux-eabi -mhard-float -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-HARDFP %s
// CHECK-ARM-HARDFP: as{{(.exe)?}}" "-mfloat-abi=hard"
//
// RUN: %clang -target aarch64-linux-gnu -mcpu=cortex-a53 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM64-MCPU %s
// CHECK-ARM64-MCPU: as{{(.exe)?}}" "-mcpu=cortex-a53"
//
// RUN: %clang -target aarch64-linux-gnu -march=armv8-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM64-MARCH %s
// CHECK-ARM64-MARCH: as{{(.exe)?}}" "-march=armv8-a"
//
// RUN: %clang -target aarch64-linux-gnu -mcpu=cortex-a53 -march=armv8-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM64-ALL %s
// CHECK-ARM64-ALL: as{{(.exe)?}}" "-march=armv8-a" "-mcpu=cortex-a53"
//
// RUN: %clang -target aarch64_be-linux-gnu -mcpu=cortex-a53 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM64-MCPU %s
//
// RUN: %clang -target aarch64_be-linux-gnu -march=armv8-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM64-MARCH %s
//
// RUN: %clang -target aarch64_be-linux-gnu -mcpu=cortex-a53 -march=armv8-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM64-ALL %s
//
// RUN: %clang -target ppc-linux -mcpu=invalid-cpu -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-PPC-NO-MCPU %s
// CHECK-PPC-NO-MCPU-NOT: as{{.*}} "-mcpu=invalid-cpu"
//
// RUN: %clang -target sparc64-linux -mcpu=invalid-cpu -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SPARCV9 %s
// CHECK-SPARCV9: as
// CHECK-SPARCV9: -64
// CHECK-SPARCV9: -Av9
// CHECK-SPARCV9-NOT: -KPIC
// CHECK-SPARCV9: -o
//
// RUN: %clang -target sparc64-linux -mcpu=invalid-cpu -### \
// RUN:   -no-integrated-as -fpic -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SPARCV9PIC %s
// CHECK-SPARCV9PIC: as
// CHECK-SPARCV9PIC: -64
// CHECK-SPARCV9PIC: -Av9
// CHECK-SPARCV9PIC: -KPIC
// CHECK-SPARCV9PIC: -o
//
// RUN: %clang -target sparc-linux -mcpu=invalid-cpu -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SPARCV8 %s
// CHECK-SPARCV8: as
// CHECK-SPARCV8: -32
// CHECK-SPARCV8: -Av8
// CHECK-SPARCV8: -o
//
// RUN: %clang -target sparcel-linux -mcpu=invalid-cpu -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SPARCV8EL %s
// CHECK-SPARCV8EL: as
// CHECK-SPARCV8EL: -32
// CHECK-SPARCV8EL: -Av8
// CHECK-SPARCV8EL: -o
//
// RUN: %clang -target s390x-linux -### -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-Z-DEFAULT-ARCH %s
// CHECK-Z-DEFAULT-ARCH: as{{.*}} "-march=z10"
//
// RUN: %clang -target s390x-linux -march=z196 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-Z-ARCH-Z196 %s
// CHECK-Z-ARCH-Z196: as{{.*}} "-march=z196"
//
// RUN: %clang -target powerpc64le-linux -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-PPC64LE %s
// CHECK-PPC64LE: as{{.*}} "-mpower8"
//
// RUN: %clang -target powerpc64-linux -mcpu=pwr7 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-PPC64 %s
// CHECK-PPC64: as{{.*}} "-mpower7"
//
// RUN: %clang -target powerpc-linux -mcpu=pwr9 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-PPC32 %s
// CHECK-PPC32: as{{.*}} "-mpower9"
