// Test that different values of -mfpu pick correct ARM FPU target-feature(s).

// RUN: %clang -target arm-linux-eabi %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-target-feature" "+vfp2"
// CHECK-DEFAULT-NOT: "-target-feature" "+vfp3"
// CHECK-DEFAULT-NOT: "-target-feature" "+d16"
// CHECK-DEFAULT-NOT: "-target-feature" "+neon"

// RUN: %clang -target arm-linux-eabi -mfpu=fpa %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// RUN: %clang -target arm-linux-eabi -mfpu=fpe2 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// RUN: %clang -target arm-linux-eabi -mfpu=fpe3 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// RUN: %clang -target arm-linux-eabi -mfpu=maverick %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// CHECK-FPA: error: {{.*}} does not support '-mfpu={{fpa|fpe|fpe2|fpe3|maverick}}'

// RUN: %clang -target arm-linux-eabi -mfpu=vfp %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP %s
// CHECK-VFP: "-target-feature" "+vfp2"
// CHECK-VFP: "-target-feature" "-vfp3"
// CHECK-VFP: "-target-feature" "-vfp4"
// CHECK-VFP: "-target-feature" "-fp-armv8"
// CHECK-VFP: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=vfp3 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3 %s
// CHECK-VFP3: "-target-feature" "+vfp3"
// CHECK-VFP3: "-target-feature" "-vfp4"
// CHECK-VFP3: "-target-feature" "-fp-armv8"
// CHECK-VFP3: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3-fp16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3-FP16 %s
// CHECK-VFP3-FP16: "-target-feature" "-fp-only-sp"
// CHECK-VFP3-FP16: "-target-feature" "-d16"
// CHECK-VFP3-FP16: "-target-feature" "+vfp3"
// CHECK-VFP3-FP16: "-target-feature" "+fp16"
// CHECK-VFP3-FP16: "-target-feature" "-vfp4"
// CHECK-VFP3-FP16: "-target-feature" "-fp-armv8"
// CHECK-VFP3-FP16: "-target-feature" "-neon"
// CHECK-VFP3-FP16: "-target-feature" "-crypto"

// RUN: %clang -target arm-linux-eabi -mfpu=vfp3-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3-D16 %s
// CHECK-VFP3-D16: "-target-feature" "-fp-only-sp"
// CHECK-VFP3-D16: "-target-feature" "+d16"
// CHECK-VFP3-D16: "-target-feature" "+vfp3"
// CHECK-VFP3-D16: "-target-feature" "-vfp4"
// CHECK-VFP3-D16: "-target-feature" "-fp-armv8"
// CHECK-VFP3-D16: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3-d16-fp16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3-D16-FP16 %s
// CHECK-VFP3-D16-FP16: "-target-feature" "-fp-only-sp"
// CHECK-VFP3-D16-FP16: "-target-feature" "+d16"
// CHECK-VFP3-D16-FP16: "-target-feature" "+vfp3"
// CHECK-VFP3-D16-FP16: "-target-feature" "+fp16"
// CHECK-VFP3-D16-FP16: "-target-feature" "-vfp4"
// CHECK-VFP3-D16-FP16: "-target-feature" "-fp-armv8"
// CHECK-VFP3-D16-FP16: "-target-feature" "-neon"
// CHECK-VFP3-D16-FP16: "-target-feature" "-crypto"

// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3xd %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3XD %s
// CHECK-VFP3XD: "-target-feature" "+fp-only-sp"
// CHECK-VFP3XD: "-target-feature" "+d16"
// CHECK-VFP3XD: "-target-feature" "+vfp3"
// CHECK-VFP3XD: "-target-feature" "-fp16"
// CHECK-VFP3XD: "-target-feature" "-vfp4"
// CHECK-VFP3XD: "-target-feature" "-fp-armv8"
// CHECK-VFP3XD: "-target-feature" "-neon"
// CHECK-VFP3XD: "-target-feature" "-crypto"

// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3xd-fp16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3XD-FP16 %s
// CHECK-VFP3XD-FP16: "-target-feature" "+fp-only-sp"
// CHECK-VFP3XD-FP16: "-target-feature" "+d16"
// CHECK-VFP3XD-FP16: "-target-feature" "+vfp3"
// CHECK-VFP3XD-FP16: "-target-feature" "+fp16"
// CHECK-VFP3XD-FP16: "-target-feature" "-vfp4"
// CHECK-VFP3XD-FP16: "-target-feature" "-fp-armv8"
// CHECK-VFP3XD-FP16: "-target-feature" "-neon"
// CHECK-VFP3XD-FP16: "-target-feature" "-crypto"

// RUN: %clang -target arm-linux-eabi -mfpu=vfp4 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP4 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv4 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP4 %s
// CHECK-VFP4: "-target-feature" "+vfp4"
// CHECK-VFP4: "-target-feature" "-fp-armv8"
// CHECK-VFP4: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=vfp4-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP4-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv4-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP4-D16 %s
// CHECK-VFP4-D16: "-target-feature" "-fp-only-sp"
// CHECK-VFP4-D16: "-target-feature" "+d16"
// CHECK-VFP4-D16: "-target-feature" "+vfp4"
// CHECK-VFP4-D16: "-target-feature" "-fp-armv8"
// CHECK-VFP4-D16: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=fp4-sp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP4-SP-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=fpv4-sp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP4-SP-D16 %s
// CHECK-FP4-SP-D16: "-target-feature" "+fp-only-sp"
// CHECK-FP4-SP-D16: "-target-feature" "+d16"
// CHECK-FP4-SP-D16: "-target-feature" "+vfp4"
// CHECK-FP4-SP-D16: "-target-feature" "-fp-armv8"
// CHECK-FP4-SP-D16: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=fp5-sp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP5-SP-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=fpv5-sp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP5-SP-D16 %s
// CHECK-FP5-SP-D16: "-target-feature" "+fp-only-sp"
// CHECK-FP5-SP-D16: "-target-feature" "+d16"
// CHECK-FP5-SP-D16: "-target-feature" "+fp-armv8"
// CHECK-FP5-SP-D16: "-target-feature" "-neon"
// CHECK-FP5-SP-D16: "-target-feature" "-crypto"

// RUN: %clang -target arm-linux-eabi -mfpu=fp5-dp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP5-DP-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=fpv5-dp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP5-DP-D16 %s
// CHECK-FP5-DP-D16: "-target-feature" "-fp-only-sp"
// CHECK-FP5-DP-D16: "-target-feature" "+d16"
// CHECK-FP5-DP-D16: "-target-feature" "+fp-armv8"
// CHECK-FP5-DP-D16: "-target-feature" "-neon"
// CHECK-FP5-DP-D16: "-target-feature" "-crypto"

// RUN: %clang -target arm-linux-eabi -mfpu=neon %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON %s
// CHECK-NEON: "-target-feature" "+neon"

// RUN: %clang -target arm-linux-eabi -mfpu=neon-fp16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON-FP16 %s
// CHECK-NEON-FP16: "-target-feature" "-fp-only-sp"
// CHECK-NEON-FP16: "-target-feature" "-d16"
// CHECK-NEON-FP16: "-target-feature" "+vfp3"
// CHECK-NEON-FP16: "-target-feature" "+fp16"
// CHECK-NEON-FP16: "-target-feature" "-vfp4"
// CHECK-NEON-FP16: "-target-feature" "-fp-armv8"
// CHECK-NEON-FP16: "-target-feature" "+neon"
// CHECK-NEON-FP16: "-target-feature" "-crypto"

// RUN: %clang -target arm-linux-eabi -mfpu=neon-vfpv3 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON-VFPV3 %s
// CHECK-NEON-VFPV3: "-target-feature" "+neon"

// RUN: %clang -target arm-linux-eabi -mfpu=neon-vfpv4 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON-VFPV4 %s
// CHECK-NEON-VFPV4: "-target-feature" "+vfp4"
// CHECK-NEON-VFPV4: "-target-feature" "+neon"

// RUN: %clang -target arm-linux-eabi -msoft-float %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-FLOAT %s
// CHECK-SOFT-FLOAT: "-target-feature" "-neon"

// RUN: %clang -target armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARMV8-DEFAULT-SOFT-FP %s
// CHECK-ARMV8-DEFAULT-SOFT-FP: "-target-feature" "-neon"
// CHECK-ARMV8-DEFAULT-SOFT-FP: "-target-feature" "-crypto"

// RUN: %clang -target armv8 -mfpu=fp-armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARMV8-SOFT-FLOAT %s
// CHECK-ARMV8-SOFT-FLOAT: "-target-feature" "+fp-armv8"
// CHECK-ARMV8-SOFT-FLOAT: "-target-feature" "-neon"
// CHECK-ARMV8-SOFT-FLOAT: "-target-feature" "-crypto"

// RUN: %clang -target armv8-linux-gnueabihf -mfpu=fp-armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP-ARMV8 %s
// CHECK-FP-ARMV8-NOT: "-target-feature" "+neon"
// CHECK-FP-ARMV8: "-target-feature" "+fp-armv8"
// CHECK-FP-ARMV8: "-target-feature" "-neon"
// CHECK-FP-ARMV8: "-target-feature" "-crypto"

// RUN: %clang -target armv8-linux-gnueabihf -mfpu=neon-fp-armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON-FP-ARMV8 %s
// CHECK-NEON-FP-ARMV8: "-target-feature" "+fp-armv8"
// CHECK-NEON-FP-ARMV8: "-target-feature" "+neon"
// CHECK-NEON-FP-ARMV8: "-target-feature" "-crypto"

// RUN: %clang -target armv8-linux-gnueabihf -mfpu=crypto-neon-fp-armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRYPTO-NEON-FP-ARMV8 %s
// CHECK-CRYPTO-NEON-FP-ARMV8: "-target-feature" "+fp-armv8"
// CHECK-CRYPTO-NEON-FP-ARMV8: "-target-feature" "+crypto"

// RUN: %clang -target armv8-linux-gnueabi -mfpu=none %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FP %s
// CHECK-NO-FP: "-target-feature" "-fp-only-sp"
// CHECK-NO-FP: "-target-feature" "-d16"
// CHECK-NO-FP: "-target-feature" "-vfp2"
// CHECK-NO-FP: "-target-feature" "-vfp3"
// CHECK-NO-FP: "-target-feature" "-vfp4"
// CHECK-NO-FP: "-target-feature" "-fp-armv8"
// CHECK-NO-FP: "-target-feature" "-neon"
// CHECK-NO-FP: "-target-feature" "-crypto"

// RUN: %clang -target arm-linux-gnueabihf %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-HF %s
// RUN: %clang -target arm-linux-musleabihf %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-HF %s
// CHECK-HF: "-target-cpu" "arm1176jzf-s"

// RUN: %clang -target armv7-apple-darwin -x assembler %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=ASM %s
// ASM-NOT: -target-feature
