// Test that different values of -mfpu pick correct ARM FPU target-feature(s).

// RUN: %clang -target arm-linux-eabi %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-target-feature" "+soft-float"
// CHECK-DEFAULT-DAG: "-target-feature" "+soft-float-abi"
// CHECK-DEFAULT-NOT: "-target-feature" "+vfp2"
// CHECK-DEFAULT-NOT: "-target-feature" "+vfp3"
// CHECK-DEFAULT-NOT: "-target-feature" "+neon"

// RUN: not %clang --target=arm-linux-eabi -mfpu=fpa %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// RUN: not %clang --target=arm-linux-eabi -mfpu=fpe2 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// RUN: not %clang --target=arm-linux-eabi -mfpu=fpe3 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// RUN: not %clang --target=arm-linux-eabi -mfpu=maverick %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// CHECK-FPA: error: {{.*}} does not support '-mfpu={{fpa|fpe|fpe2|fpe3|maverick}}'

// RUN: %clang -target arm-linux-eabi -mfpu=vfp %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfp %s -mfloat-abi=soft -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-2 %s
// CHECK-VFP-NOT: "-target-feature" "+soft-float"
// CHECK-VFP-DAG: "-target-feature" "+soft-float-abi"
// CHECK-VFP-DAG: "-target-feature" "+vfp2"
// CHECK-VFP-DAG: "-target-feature" "-vfp3d16sp"
// CHECK-VFP-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-VFP-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-VFP-DAG: "-target-feature" "-neon"
// CHECK-SOFT-ABI-FP-2-DAG: "-target-feature" "+soft-float-abi"
// CHECK-SOFT-ABI-FP-2-DAG: "-target-feature" "-vfp3d16sp"
// CHECK-SOFT-ABI-FP-2-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-SOFT-ABI-FP-2-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-SOFT-ABI-FP-2-DAG: "-target-feature" "-neon"
// CHECK-SOFT-ABI-FP-2-DAG: "-target-feature" "-sha2"
// CHECK-SOFT-ABI-FP-2-DAG: "-target-feature" "-aes"
// CHECK-SOFT-ABI-FP-2-DAG: "-target-feature" "-vfp2sp"

// RUN: %clang -target arm-linux-eabi -mfpu=vfp3 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3 -mfloat-abi=soft %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-3 %s
// CHECK-VFP3-NOT: "-target-feature" "+soft-float"
// CHECK-VFP3-DAG: "-target-feature" "+soft-float-abi"
// CHECK-VFP3-DAG: "-target-feature" "+vfp3"
// CHECK-VFP3-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-VFP3-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-VFP3-DAG: "-target-feature" "-neon"
// CHECK-SOFT-ABI-FP-3-DAG: "-target-feature" "+soft-float-abi"
// CHECK-SOFT-ABI-FP-3-DAG: "-target-feature" "-vfp2sp"
// CHECK-SOFT-ABI-FP-3-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-SOFT-ABI-FP-3-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-SOFT-ABI-FP-3-DAG: "-target-feature" "-neon"
// CHECK-SOFT-ABI-FP-3-DAG: "-target-feature" "-sha2"
// CHECK-SOFT-ABI-FP-3-DAG: "-target-feature" "-aes"
// CHECK-SOFT-ABI-FP-3-DAG: "-target-feature" "-vfp3d16sp"

// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3-fp16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3-FP16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3-fp16 -mfloat-abi=soft %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-3 %s
// CHECK-VFP3-FP16-NOT: "-target-feature" "+soft-float"
// CHECK-VFP3-FP16-DAG: "-target-feature" "+soft-float-abi"
// CHECK-VFP3-FP16-DAG: "-target-feature" "+vfp3"
// CHECK-VFP3-FP16-DAG: "-target-feature" "+fp16"
// CHECK-VFP3-FP16-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-VFP3-FP16-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-VFP3-FP16-DAG: "-target-feature" "+fp64"
// CHECK-VFP3-FP16-DAG: "-target-feature" "+d32"
// CHECK-VFP3-FP16-DAG: "-target-feature" "-neon"
// CHECK-VFP3-FP16-DAG: "-target-feature" "-sha2"
// CHECK-VFP3-FP16-DAG: "-target-feature" "-aes"

// RUN: %clang -target arm-linux-eabi -mfpu=vfp3-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3-d16 -mfloat-abi=soft %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-3 %s
// CHECK-VFP3-D16-NOT: "-target-feature" "+soft-float"
// CHECK-VFP3-D16-DAG: "-target-feature" "+soft-float-abi"
// CHECK-VFP3-D16-DAG: "-target-feature" "+vfp3d16"
// CHECK-VFP3-D16-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-VFP3-D16-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-VFP3-D16-DAG: "-target-feature" "+fp64"
// CHECK-VFP3-D16-DAG: "-target-feature" "-d32"
// CHECK-VFP3-D16-DAG: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3-d16-fp16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3-D16-FP16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3-d16-fp16 -mfloat-abi=soft %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-3 %s
// CHECK-VFP3-D16-FP16-NOT: "-target-feature" "+soft-float"
// CHECK-VFP3-D16-FP16-DAG: "-target-feature" "+soft-float-abi"
// CHECK-VFP3-D16-FP16-DAG: "-target-feature" "+vfp3d16"
// CHECK-VFP3-D16-FP16-DAG: "-target-feature" "+fp16"
// CHECK-VFP3-D16-FP16-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-VFP3-D16-FP16-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-VFP3-D16-FP16-DAG: "-target-feature" "+fp64"
// CHECK-VFP3-D16-FP16-DAG: "-target-feature" "-d32"
// CHECK-VFP3-D16-FP16-DAG: "-target-feature" "-neon"
// CHECK-VFP3-D16-FP16-DAG: "-target-feature" "-sha2"
// CHECK-VFP3-D16-FP16-DAG: "-target-feature" "-aes"

// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3xd %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3XD %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3xd -mfloat-abi=soft %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-3 %s
// CHECK-VFP3XD-NOT: "-target-feature" "+soft-float"
// CHECK-VFP3XD-DAG: "-target-feature" "+soft-float-abi"
// CHECK-VFP3XD-DAG: "-target-feature" "-fp64"
// CHECK-VFP3XD-DAG: "-target-feature" "-d32"
// CHECK-VFP3XD-DAG: "-target-feature" "+vfp3d16sp"
// CHECK-VFP3XD-DAG: "-target-feature" "-fp16"
// CHECK-VFP3XD-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-VFP3XD-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-VFP3XD-DAG: "-target-feature" "-neon"
// CHECK-VFP3XD-DAG: "-target-feature" "-sha2"
// CHECK-VFP3XD-DAG: "-target-feature" "-aes"

// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3xd-fp16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3XD-FP16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3xd-fp16 -mfloat-abi=soft %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-3 %s
// CHECK-VFP3XD-FP16-NOT: "-target-feature" "+soft-float"
// CHECK-VFP3XD-FP16-DAG: "-target-feature" "+soft-float-abi"
// CHECK-VFP3XD-FP16-DAG: "-target-feature" "+vfp3d16sp"
// CHECK-VFP3XD-FP16-DAG: "-target-feature" "+fp16"
// CHECK-VFP3XD-FP16-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-VFP3XD-FP16-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-VFP3XD-FP16-DAG: "-target-feature" "-fp64"
// CHECK-VFP3XD-FP16-DAG: "-target-feature" "-d32"
// CHECK-VFP3XD-FP16-DAG: "-target-feature" "-neon"
// CHECK-VFP3XD-FP16-DAG: "-target-feature" "-sha2"
// CHECK-VFP3XD-FP16-DAG: "-target-feature" "-aes"

// RUN: %clang -target arm-linux-eabi -mfpu=vfp4 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP4 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv4 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP4 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv4 -mfloat-abi=soft %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-4 %s
// CHECK-VFP4-NOT: "-target-feature" "+soft-float"
// CHECK-VFP4-DAG: "-target-feature" "+soft-float-abi"
// CHECK-VFP4-DAG: "-target-feature" "+vfp4"
// CHECK-VFP4-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-VFP4-DAG: "-target-feature" "-neon"
// CHECK-SOFT-ABI-FP-4-DAG: "-target-feature" "+soft-float-abi"
// CHECK-SOFT-ABI-FP-4-DAG: "-target-feature" "-vfp2sp"
// CHECK-SOFT-ABI-FP-4-DAG: "-target-feature" "-vfp3d16sp"
// CHECK-SOFT-ABI-FP-4-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-SOFT-ABI-FP-4-DAG: "-target-feature" "-neon"
// CHECK-SOFT-ABI-FP-4-DAG: "-target-feature" "-sha2"
// CHECK-SOFT-ABI-FP-4-DAG: "-target-feature" "-aes"
// CHECK-SOFT-ABI-FP-4-DAG: "-target-feature" "-vfp4d16sp"

// RUN: %clang -target arm-linux-eabi -mfpu=vfp4-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP4-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv4-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP4-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv4-d16 -mfloat-abi=soft %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-4 %s
// CHECK-VFP4-D16-NOT: "-target-feature" "+soft-float"
// CHECK-VFP4-D16-DAG: "-target-feature" "+soft-float-abi"
// CHECK-VFP4-D16-DAG: "-target-feature" "+vfp4d16"
// CHECK-VFP4-D16-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-VFP4-D16-DAG: "-target-feature" "+fp64"
// CHECK-VFP4-D16-DAG: "-target-feature" "-d32"
// CHECK-VFP4-D16-DAG: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=fp4-sp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP4-SP-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=fpv4-sp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP4-SP-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=fpv4-sp-d16 -mfloat-abi=soft %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-4 %s
// CHECK-FP4-SP-D16-NOT: "-target-feature" "+soft-float"
// CHECK-FP4-SP-D16-DAG: "-target-feature" "+soft-float-abi"
// CHECK-FP4-SP-D16-DAG: "-target-feature" "+vfp4d16sp"
// CHECK-FP4-SP-D16-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-FP4-SP-D16-DAG: "-target-feature" "-fp64"
// CHECK-FP4-SP-D16-DAG: "-target-feature" "-d32"
// CHECK-FP4-SP-D16-DAG: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=fp5-sp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP5-SP-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=fpv5-sp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP5-SP-D16 %s
// RUN: not %clang --target=arm-linux-eabi -mfpu=fp-armv8-sp-d16 -mfloat-abi=soft %s -### -o %t.o \
// RUN:   2>&1 | FileCheck --check-prefix=CHECK-SOFT-ABI-FP %s
// CHECK-FP5-SP-D16-NOT: "-target-feature" "+soft-float"
// CHECK-FP5-SP-D16-DAG: "-target-feature" "+soft-float-abi"
// CHECK-FP5-SP-D16-DAG: "-target-feature" "+fp-armv8d16sp"
// CHECK-FP5-SP-D16-DAG: "-target-feature" "-neon"
// CHECK-FP5-SP-D16-DAG: "-target-feature" "-fp64"
// CHECK-FP5-SP-D16-DAG: "-target-feature" "-d32"
// CHECK-FP5-SP-D16-DAG: "-target-feature" "-sha2"
// CHECK-FP5-SP-D16-DAG: "-target-feature" "-aes"

// RUN: %clang -target arm-linux-eabi -mfpu=fp5-dp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP5-DP-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=fpv5-dp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP5-DP-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=fpv5-dp-d16 %s -mfloat-abi=soft -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-5 %s
// CHECK-FP5-DP-D16-NOT: "-target-feature" "+soft-float"
// CHECK-FP5-DP-D16-DAG: "-target-feature" "+soft-float-abi"
// CHECK-FP5-DP-D16-DAG: "-target-feature" "+fp-armv8d16"
// CHECK-FP5-DP-D16-DAG: "-target-feature" "+fp64"
// CHECK-FP5-DP-D16-DAG: "-target-feature" "-d32"
// CHECK-FP5-DP-D16-DAG: "-target-feature" "-neon"
// CHECK-FP5-DP-D16-DAG: "-target-feature" "-sha2"
// CHECK-FP5-DP-D16-DAG: "-target-feature" "-aes"
// CHECK-SOFT-ABI-FP-5-DAG: "-target-feature" "+soft-float"
// CHECK-SOFT-ABI-FP-5-DAG: "-target-feature" "+soft-float-abi"
// CHECK-SOFT-ABI-FP-5-DAG: "-target-feature" "-vfp2sp"
// CHECK-SOFT-ABI-FP-5-DAG: "-target-feature" "-vfp3d16sp"
// CHECK-SOFT-ABI-FP-5-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-SOFT-ABI-FP-5-DAG: "-target-feature" "-neon"
// CHECK-SOFT-ABI-FP-5-DAG: "-target-feature" "-sha2"
// CHECK-SOFT-ABI-FP-5-DAG: "-target-feature" "-aes"
// CHECK-SOFT-ABI-FP-5-DAG: "-target-feature" "-fp-armv8d16sp"

// RUN: %clang -target arm-linux-eabi -mfpu=neon %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON %s
// RUN: %clang -target arm-linux-eabi -mfpu=neon -mfloat-abi=soft %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-6 %s
// CHECK-NEON-NOT: "-target-feature" "+soft-float"
// CHECK-NEON-DAG: "-target-feature" "+neon"
// CHECK-SOFT-ABI-FP-6-DAG: "-target-feature" "+soft-float-abi"
// CHECK-SOFT-ABI-FP-6-DAG: "-target-feature" "-vfp2sp"
// CHECK-SOFT-ABI-FP-6-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-SOFT-ABI-FP-6-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-SOFT-ABI-FP-6-DAG: "-target-feature" "-sha2"
// CHECK-SOFT-ABI-FP-6-DAG: "-target-feature" "-aes"
// CHECK-SOFT-ABI-FP-6-DAG: "-target-feature" "-vfp3d16sp"
// CHECK-SOFT-ABI-FP-6-DAG: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=neon-fp16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON-FP16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=neon-fp16 -mfloat-abi=soft %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-6 %s
// CHECK-NEON-FP16-NOT: "-target-feature" "+soft-float"
// CHECK-NEON-FP16-DAG: "-target-feature" "+soft-float-abi"
// CHECK-NEON-FP16-DAG: "-target-feature" "+vfp3"
// CHECK-NEON-FP16-DAG: "-target-feature" "+fp16"
// CHECK-NEON-FP16-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-NEON-FP16-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-NEON-FP16-DAG: "-target-feature" "+fp64"
// CHECK-NEON-FP16-DAG: "-target-feature" "+d32"
// CHECK-NEON-FP16-DAG: "-target-feature" "+neon"
// CHECK-NEON-FP16-DAG: "-target-feature" "-sha2"
// CHECK-NEON-FP16-DAG: "-target-feature" "-aes"

// RUN: %clang -target arm-linux-eabi -mfpu=neon-vfpv3 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON-VFPV3 %s
// RUN: %clang -target arm-linux-eabi -mfpu=neon-vfpv3 -mfloat-abi=soft %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-6 %s
// CHECK-NEON-VFPV3-NOT: "-target-feature" "+soft-float"
// CHECK-NEON-VFPV3-DAG: "-target-feature" "+soft-float-abi"
// CHECK-NEON-VFPV3-DAG: "-target-feature" "+vfp3"
// CHECK-NEON-VFPV3-DAG: "-target-feature" "+neon"

// RUN: %clang -target arm-linux-eabi -mfpu=neon-vfpv4 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON-VFPV4 %s
// RUN: %clang -target arm-linux-eabi -mfpu=neon-vfpv4 -mfloat-abi=soft %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-7 %s
// CHECK-NEON-VFPV4-NOT: "-target-feature" "+soft-float"
// CHECK-NEON-VFPV4-DAG: "-target-feature" "+soft-float-abi"
// CHECK-NEON-VFPV4-DAG: "-target-feature" "+vfp4"
// CHECK-NEON-VFPV4-DAG: "-target-feature" "+neon"
// CHECK-SOFT-ABI-FP-7-DAG: "-target-feature" "+soft-float-abi"
// CHECK-SOFT-ABI-FP-7-DAG: "-target-feature" "-vfp2sp"
// CHECK-SOFT-ABI-FP-7-DAG: "-target-feature" "-vfp3d16sp"
// CHECK-SOFT-ABI-FP-7-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-SOFT-ABI-FP-7-DAG: "-target-feature" "-sha2"
// CHECK-SOFT-ABI-FP-7-DAG: "-target-feature" "-aes"
// CHECK-SOFT-ABI-FP-7-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-SOFT-ABI-FP-7-DAG: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -msoft-float %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP %s
// RUN: %clang -target armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP %s
// RUN: %clang -target armv8a -mfpu=neon %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP-8 %s
// CHECK-SOFT-ABI-FP-8-DAG: "-target-feature" "+soft-float-abi"
// CHECK-SOFT-ABI-FP-8-DAG: "-target-feature" "-vfp2sp"
// CHECK-SOFT-ABI-FP-8-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-SOFT-ABI-FP-8-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-SOFT-ABI-FP-8-DAG: "-target-feature" "-sha2"
// CHECK-SOFT-ABI-FP-8-DAG: "-target-feature" "-aes"
// CHECK-SOFT-ABI-FP-8-DAG: "-target-feature" "-vfp3d16sp"
// CHECK-SOFT-ABI-FP-8-DAG: "-target-feature" "-neon"

// RUN: %clang -target armv8 -mfpu=fp-armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARMV8-SOFT-FLOAT %s
// CHECK-ARMV8-SOFT-FLOAT-DAG: "-target-feature" "+soft-float"
// CHECK-ARMV8-SOFT-FLOAT-DAG: "-target-feature" "+soft-float-abi"
// NOT-CHECK-ARMV8-SOFT-FLOAT: "-target-feature" "+fp-armv8"
// CHECK-ARMV9-SOFT-FLOAT-DAG: "-target-feature" "-neon"
// CHECK-ARMV8-SOFT-FLOAT-DAG: "-target-feature" "-sha2"
// CHECK-ARMV8-SOFT-FLOAT-DAG: "-target-feature" "-aes"

// RUN: %clang -target armv8-linux-gnueabihf -mfpu=fp-armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP-ARMV8 %s
// CHECK-FP-ARMV8-NOT: "-target-feature" "+soft-float"
// CHECK-FP-ARMV8-NOT: "-target-feature" "+soft-float-abi"
// CHECK-FP-ARMV8-DAG: "-target-feature" "+fp-armv8"
// CHECK-FP-ARMV8-DAG: "-target-feature" "-neon"
// CHECK-FP-ARMV8-DAG: "-target-feature" "-sha2"
// CHECK-FP-ARMV8-DAG: "-target-feature" "-aes"

// RUN: %clang -target armv8-linux-gnueabihf -mfpu=neon-fp-armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON-FP-ARMV8 %s
// CHECK-NEON-FP-ARMV8-NOT: "-target-feature" "+soft-float"
// CHECK-NEON-FP-ARMV8-NOT: "-target-feature" "+soft-float-abi"
// CHECK-NEON-FP-ARMV8-DAG: "-target-feature" "+fp-armv8"
// CHECK-NEON-FP-ARMV8-DAG: "-target-feature" "+neon"
// CHECK-NEON-FP-ARMV8-DAG: "-target-feature" "-sha2"
// CHECK-NEON-FP-ARMV8-DAG: "-target-feature" "-aes"

// RUN: %clang -target armv8-linux-gnueabihf -mfpu=crypto-neon-fp-armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRYPTO-NEON-FP-ARMV8 %s
// CHECK-CRYPTO-NEON-FP-ARMV8-NOT: "-target-feature" "+soft-float"
// CHECK-CRYPTO-NEON-FP-ARMV8-NOT: "-target-feature" "+soft-float-abi"
// CHECK-CRYPTO-NEON-FP-ARMV8-DAG: "-target-feature" "+fp-armv8"
// CHECK-CRYPTO-NEON-FP-ARMV8-DAG: "-target-feature" "+sha2"
// CHECK-CRYPTO-NEON-FP-ARMV8-DAG: "-target-feature" "+aes"

// RUN: %clang -target armv8-linux-gnueabi -mfpu=none %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FP %s
// CHECK-NO-FP-NOT: "-target-feature" "+soft-float"
// CHECK-NO-FP-DAG: "-target-feature" "+soft-float-abi"
// CHECK-NO-FP-DAG: "-target-feature" "-fpregs"
// CHECK-NO-FP-DAG: "-target-feature" "-vfp2sp"
// CHECK-NO-FP-DAG: "-target-feature" "-vfp3d16sp"
// CHECK-NO-FP-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-NO-FP-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-NO-FP-DAG: "-target-feature" "-fp64"
// CHECK-NO-FP-DAG: "-target-feature" "-d32"
// CHECK-NO-FP-DAG: "-target-feature" "-neon"
// CHECK-NO-FP-DAG: "-target-feature" "-sha2"
// CHECK-NO-FP-DAG: "-target-feature" "-aes"

// RUN: %clang -target arm-linux-gnueabihf %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-HF %s
// RUN: %clang -target arm-linux-musleabihf %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-HF %s
// CHECK-HF-NOT: "-target-feature" "+soft-float"
// CHECK-HF-NOT: "-target-feature" "+soft-float-abi"
// CHECK-HF-DAG: "-target-cpu" "arm1176jzf-s"

// RUN: %clang -target armv7-apple-darwin -x assembler %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=ASM %s
// ASM-NOT: -target-feature

// RUN: %clang -target armv8-linux-gnueabi -mfloat-abi=soft -mfpu=none %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP %s
// RUN: %clang -target armv7-linux-gnueabi -mfloat-abi=soft -mfpu=none %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP %s
// RUN: %clang -target armv6-linux-gnueabi -mfloat-abi=soft -mfpu=none %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP %s
// RUN: %clang -target armv5-linux-gnueabi -mfloat-abi=soft -mfpu=none %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP %s
// RUN: %clang -target armv4-linux-gnueabi -mfloat-abi=soft -mfpu=none %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP %s
// RUN: %clang -target armv8-linux-gnueabi -msoft-float -mfpu=none %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP %s
// RUN: %clang -target armv8-linux-gnueabi -mfloat-abi=soft %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP %s
// RUN: %clang -target armv8-linux-gnueabi -msoft-float %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-ABI-FP %s
// CHECK-SOFT-ABI-FP-DAG: "-target-feature" "+soft-float"
// CHECK-SOFT-ABI-FP-DAG: "-target-feature" "+soft-float-abi"
// CHECK-SOFT-ABI-FP-DAG: "-target-feature" "-vfp2sp"
// CHECK-SOFT-ABI-FP-DAG: "-target-feature" "-vfp3d16sp"
// CHECK-SOFT-ABI-FP-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-SOFT-ABI-FP-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-SOFT-ABI-FP-DAG: "-target-feature" "-neon"
// CHECK-SOFT-ABI-FP-DAG: "-target-feature" "-sha2"
// CHECK-SOFT-ABI-FP-DAG: "-target-feature" "-aes"
// CHECK-SOFT-ABI-FP-DAG: "-target-feature" "-fpregs"

// RUN: %clang -target arm-linux-androideabi21 %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARM5-ANDROID-FP-DEFAULT %s
// CHECK-ARM5-ANDROID-FP-DEFAULT-DAG: "-target-feature" "+soft-float"
// CHECK-ARM5-ANDROID-FP-DEFAULT-DAG: "-target-feature" "+soft-float-abi"
// CHECK-ARM5-ANDROID-FP-DEFAULT-NOT: "-target-feature" "+d32"
// CHECK-ARM5-ANDROID-FP-DEFAULT-NOT: "-target-feature" "+vfp3"
// CHECK-ARM5-ANDROID-FP-DEFAULT-NOT: "-target-feature" "+vfp4"
// CHECK-ARM5-ANDROID-FP-DEFAULT-NOT: "-target-feature" "+fp-armv8"
// CHECK-ARM5-ANDROID-FP-DEFAULT-NOT: "-target-feature" "+neon"
// CHECK-ARM5-ANDROID-FP-DEFAULT-NOT: "-target-feature" "+sha2"
// CHECK-ARM5-ANDROID-FP-DEFAULT-NOT: "-target-feature" "+aes"

// RUN: %clang -target armv7-linux-androideabi21 %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARM7-ANDROID-FP-DEFAULT %s
// CHECK-ARM7-ANDROID-FP-DEFAULT-NOT: "-target-feature" "+soft-float"
// CHECK-ARM7-ANDROID-FP-DEFAULT-DAG: "-target-feature" "+soft-float-abi"
// CHECK-ARM7-ANDROID-FP-DEFAULT-DAG: "-target-feature" "+vfp3"
// CHECK-ARM7-ANDROID-FP-DEFAULT-DAG: "-target-feature" "-vfp4"
// CHECK-ARM7-ANDROID-FP-DEFAULT-DAG: "-target-feature" "-fp-armv8"
// CHECK-ARM7-ANDROID-FP-DEFAULT-DAG: "-target-feature" "+neon"
// CHECK-ARM7-ANDROID-FP-DEFAULT-NOT: "-target-feature" "+sha2"
// CHECK-ARM7-ANDROID-FP-DEFAULT-NOT: "-target-feature" "+aes"

// RUN: %clang -target armv7-linux-androideabi21 %s -mfpu=vfp3-d16 -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARM7-ANDROID-FP-D16 %s
// CHECK-ARM7-ANDROID-FP-D16-NOT: "-target-feature" "+soft-float"
// CHECK-ARM7-ANDROID-FP-D16-DAG: "-target-feature" "+soft-float-abi"
// CHECK-ARM7-ANDROID-FP-D16-DAG: "-target-feature" "-d32"
// CHECK-ARM7-ANDROID-FP-D16-DAG: "-target-feature" "+vfp3d16"
// CHECK-ARM7-ANDROID-FP-D16-NOT: "-target-feature" "+vfp4"
// CHECK-ARM7-ANDROID-FP-D16-NOT: "-target-feature" "+fp-armv8"
// CHECK-ARM7-ANDROID-FP-D16-NOT: "-target-feature" "+neon"
// CHECK-ARM7-ANDROID-FP-D16-NOT: "-target-feature" "+sha2"
// CHECK-ARM7-ANDROID-FP-D16-NOT: "-target-feature" "+aes"

// RUN: %clang -target arm-none-none-eabi %s -march=armv8.1-m.main+mve.fp+fp.dp -mfloat-abi=soft -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFTFLOATABI-INHIBITS-MVE %s
// CHECK-SOFTFLOATABI-INHIBITS-MVE-NOT: "-target-feature" "+mve"
// CHECK-SOFTFLOATABI-INHIBITS-MVE-DAG: "-target-feature" "-mve"
// CHECK-SOFTFLOATABI-INHIBITS-MVE-DAG: "-target-feature" "-mve.fp"

// RUN: %clang -target arm-none-none-eabi %s -march=armv8.1-m.main+mve.fp -mfpu=none -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MVEFP-FPUNONE %s
// CHECK-MVEFP-FPUNONE-DAG: "-target-feature" "-vfp2sp"
// CHECK-MVEFP-FPUNONE-DAG: "-target-feature" "-vfp3d16sp"
// CHECK-MVEFP-FPUNONE-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-MVEFP-FPUNONE-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-MVEFP-FPUNONE-DAG: "-target-feature" "-fp64"
// CHECK-MVEFP-FPUNONE-DAG: "-target-feature" "-d32"
// CHECK-MVEFP-FPUNONE-DAG: "-target-feature" "-neon"
// CHECK-MVEFP-FPUNONE-DAG: "-target-feature" "-sha2"
// CHECK-MVEFP-FPUNONE-DAG: "-target-feature" "-aes"
// CHECK-MVEFP-FPUNONE-DAG: "-target-feature" "+mve"
// CHECK-MVEFP-FPUNONE-DAG: "-target-feature" "+dsp"
// CHECK-MVEFP-FPUNONE-DAG: "-target-feature" "-mve.fp"
// CHECK-MVEFP-FPUNONE-NOT: "-target-feature" "-fpregs"

// RUN: %clang -target arm-none-none-eabi %s -march=armv8.1-m.main+mve.fp+nomve -mfpu=none -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MVEFP-NOMVE-FPUNONE %s
// CHECK-MVEFP-NOMVE-FPUNONE-DAG: "-target-feature" "-vfp2sp"
// CHECK-MVEFP-NOMVE-FPUNONE-DAG: "-target-feature" "-vfp3d16sp"
// CHECK-MVEFP-NOMVE-FPUNONE-DAG: "-target-feature" "-vfp4d16sp"
// CHECK-MVEFP-NOMVE-FPUNONE-DAG: "-target-feature" "-fp-armv8d16sp"
// CHECK-MVEFP-NOMVE-FPUNONE-DAG: "-target-feature" "-fp64"
// CHECK-MVEFP-NOMVE-FPUNONE-DAG: "-target-feature" "-d32"
// CHECK-MVEFP-NOMVE-FPUNONE-DAG: "-target-feature" "-neon"
// CHECK-MVEFP-NOMVE-FPUNONE-DAG: "-target-feature" "-sha2"
// CHECK-MVEFP-NOMVE-FPUNONE-DAG: "-target-feature" "-aes"
// CHECK-MVEFP-NOMVE-FPUNONE-DAG: "-target-feature" "+dsp"
// CHECK-MVEFP-NOMVE-FPUNONE-DAG: "-target-feature" "-mve"
// CHECK-MVEFP-NOMVE-FPUNONE-DAG: "-target-feature" "-mve.fp"
// CHECK-MVEFP-NOMVE-FPUNONE-DAG: "-target-feature" "-fpregs"

// RUN: %clang -target arm-none-none-eabi %s -march=armv8.1-m.main+mve -mfpu=none -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MVEI-FPUNONE %s
// CHECK-MVEI-FPUNONE-DAG: "-target-feature" "-mve.fp"
// CHECK-MVEI-FPUNONE-DAG: "-target-feature" "+mve"
// CHECK-MVEI-FPUNONE-DAG: "-target-feature" "+dsp"
// CHECK-MVEI-FPUNONE-NOT: "-target-feature" "-fpregs"
