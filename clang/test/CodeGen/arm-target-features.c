// REQUIRES: arm-registered-target

// RUN: %clang_cc1 -triple thumbv7-linux-gnueabihf -target-cpu cortex-a8 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP3
// CHECK-VFP3: "target-features"="+dsp,+neon,+thumb-mode


// RUN: %clang_cc1 -triple thumbv7-linux-gnueabihf -target-cpu cortex-a5 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4
// CHECK-VFP4: "target-features"="+dsp,+neon,+thumb-mode,+vfp4"


// RUN: %clang_cc1 -triple thumbv7-linux-gnueabihf -target-cpu cortex-a7 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4-DIV
// RUN: %clang_cc1 -triple thumbv7-linux-gnueabi -target-cpu cortex-a12 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4-DIV
// RUN: %clang_cc1 -triple thumbv7s-linux-gnueabi -target-cpu swift -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4-DIV
// RUN: %clang_cc1 -triple thumbv7-linux-gnueabihf -target-cpu krait -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4-DIV
// CHECK-VFP4-DIV: "target-features"="+dsp,+hwdiv,+hwdiv-arm,+neon,+thumb-mode,+vfp4"

// RUN: %clang_cc1 -triple armv7-linux-gnueabihf -target-cpu cortex-a15 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4-DIV-ARM
// RUN: %clang_cc1 -triple armv7-linux-gnueabihf -target-cpu cortex-a17 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4-DIV-ARM
// CHECK-VFP4-DIV-ARM: "target-features"="+dsp,+hwdiv,+hwdiv-arm,+neon,+vfp4,-thumb-mode"

// RUN: %clang_cc1 -triple thumbv7s-apple-ios7.0 -target-cpu cyclone -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a32 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a35 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a57 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a72 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a73 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu exynos-m1 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu exynos-m2 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu exynos-m3 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// CHECK-BASIC-V8: "target-features"="+crc,+crypto,+dsp,+fp-armv8,+hwdiv,+hwdiv-arm,+neon,+thumb-mode"

// RUN: %clang_cc1 -triple armv8-linux-gnueabi -target-cpu cortex-a53 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8-ARM
// CHECK-BASIC-V8-ARM: "target-features"="+crc,+crypto,+dsp,+fp-armv8,+hwdiv,+hwdiv-arm,+neon,-thumb-mode"

// RUN: %clang_cc1 -triple thumbv7-linux-gnueabi -target-cpu cortex-r5 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP3-D16-DIV
// CHECK-VFP3-D16-DIV: "target-features"="+d16,+dsp,+hwdiv,+hwdiv-arm,+thumb-mode,+vfp3"


// RUN: %clang_cc1 -triple armv7-linux-gnueabi -target-cpu cortex-r4f -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP3-D16-THUMB-DIV
// CHECK-VFP3-D16-THUMB-DIV: "target-features"="+d16,+dsp,+hwdiv,+vfp3,-thumb-mode"


// RUN: %clang_cc1 -triple thumbv7-linux-gnueabi -target-cpu cortex-r7 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP3-D16-FP16-DIV
// RUN: %clang_cc1 -triple thumbv7-linux-gnueabi -target-cpu cortex-r8 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP3-D16-FP16-DIV
// CHECK-VFP3-D16-FP16-DIV: "target-features"="+d16,+dsp,+fp16,+hwdiv,+hwdiv-arm,+thumb-mode,+vfp3"


// RUN: %clang_cc1 -triple thumbv7-linux-gnueabi -target-cpu cortex-m4 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4-D16-SP-THUMB-DIV
// CHECK-VFP4-D16-SP-THUMB-DIV: "target-features"="+d16,+dsp,+fp-only-sp,+hwdiv,+thumb-mode,+vfp4"


// RUN: %clang_cc1 -triple thumbv7-linux-gnueabi -target-cpu cortex-m7 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP5-D16-THUMB-DIV
// CHECK-VFP5-D16-THUMB-DIV: "target-features"="+d16,+dsp,+fp-armv8,+hwdiv,+thumb-mode"


// RUN: %clang_cc1 -triple armv7-linux-gnueabi -target-cpu cortex-r4 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-THUMB-DIV
// CHECK-THUMB-DIV: "target-features"="+dsp,+hwdiv,-thumb-mode"

// RUN: %clang_cc1 -triple thumbv7-linux-gnueabi -target-cpu cortex-m3 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-THUMB-DIV-M3
// CHECK-THUMB-DIV-M3: "target-features"="+hwdiv,+thumb-mode"


void foo() {}
