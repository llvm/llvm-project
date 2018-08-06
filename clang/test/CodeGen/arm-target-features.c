// REQUIRES: arm-registered-target

// RUN: %clang_cc1 -triple thumbv7-linux-gnueabihf -target-cpu cortex-a8 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP3
// CHECK-VFP3: "target-features"="+armv7-a,+dsp,+neon,+thumb-mode,+vfp3"


// RUN: %clang_cc1 -triple thumbv7-linux-gnueabihf -target-cpu cortex-a5 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4
// CHECK-VFP4: "target-features"="+armv7-a,+dsp,+neon,+thumb-mode,+vfp4"


// RUN: %clang_cc1 -triple thumbv7-linux-gnueabihf -target-cpu cortex-a7 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4-DIV
// RUN: %clang_cc1 -triple thumbv7-linux-gnueabi -target-cpu cortex-a12 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4-DIV
// RUN: %clang_cc1 -triple thumbv7s-linux-gnueabi -target-cpu swift -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4-DIV-2
// RUN: %clang_cc1 -triple thumbv7-linux-gnueabihf -target-cpu krait -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4-DIV
// CHECK-VFP4-DIV: "target-features"="+armv7-a,+dsp,+hwdiv,+hwdiv-arm,+neon,+thumb-mode,+vfp4"
// CHECK-VFP4-DIV-2: "target-features"="+armv7s,+dsp,+hwdiv,+hwdiv-arm,+neon,+thumb-mode,+vfp4"

// RUN: %clang_cc1 -triple armv7-linux-gnueabihf -target-cpu cortex-a15 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4-DIV-ARM
// RUN: %clang_cc1 -triple armv7-linux-gnueabihf -target-cpu cortex-a17 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4-DIV-ARM
// CHECK-VFP4-DIV-ARM: "target-features"="+armv7-a,+dsp,+hwdiv,+hwdiv-arm,+neon,+vfp4,-thumb-mode"

// RUN: %clang_cc1 -triple thumbv7s-apple-ios7.0 -target-cpu cyclone -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a32 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a35 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a57 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a72 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a73 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu exynos-m1 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu exynos-m2 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu exynos-m3 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu exynos-m4 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8
// CHECK-BASIC-V8: "target-features"="+armv8-a,+crc,+crypto,+dsp,+fp-armv8,+hwdiv,+hwdiv-arm,+neon,+thumb-mode"

// RUN: %clang_cc1 -triple armv8-linux-gnueabi -target-cpu cortex-a53 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-BASIC-V8-ARM
// CHECK-BASIC-V8-ARM: "target-features"="+armv8-a,+crc,+crypto,+dsp,+fp-armv8,+hwdiv,+hwdiv-arm,+neon,-thumb-mode"

// RUN: %clang_cc1 -triple thumbv7-linux-gnueabi -target-cpu cortex-r5 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP3-D16-DIV
// CHECK-VFP3-D16-DIV: "target-features"="+armv7-r,+d16,+dsp,+hwdiv,+hwdiv-arm,+thumb-mode,+vfp3"


// RUN: %clang_cc1 -triple armv7-linux-gnueabi -target-cpu cortex-r4f -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP3-D16-THUMB-DIV
// CHECK-VFP3-D16-THUMB-DIV: "target-features"="+armv7-r,+d16,+dsp,+hwdiv,+vfp3,-thumb-mode"


// RUN: %clang_cc1 -triple thumbv7-linux-gnueabi -target-cpu cortex-r7 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP3-D16-FP16-DIV
// RUN: %clang_cc1 -triple thumbv7-linux-gnueabi -target-cpu cortex-r8 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP3-D16-FP16-DIV
// CHECK-VFP3-D16-FP16-DIV: "target-features"="+armv7-r,+d16,+dsp,+fp16,+hwdiv,+hwdiv-arm,+thumb-mode,+vfp3"


// RUN: %clang_cc1 -triple thumbv7-linux-gnueabi -target-cpu cortex-m4 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP4-D16-SP-THUMB-DIV
// CHECK-VFP4-D16-SP-THUMB-DIV: "target-features"="+armv7e-m,+d16,+dsp,+fp-only-sp,+hwdiv,+thumb-mode,+vfp4"


// RUN: %clang_cc1 -triple thumbv7-linux-gnueabi -target-cpu cortex-m7 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-VFP5-D16-THUMB-DIV
// CHECK-VFP5-D16-THUMB-DIV: "target-features"="+armv7e-m,+d16,+dsp,+fp-armv8,+hwdiv,+thumb-mode"


// RUN: %clang_cc1 -triple armv7-linux-gnueabi -target-cpu cortex-r4 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-THUMB-DIV
// CHECK-THUMB-DIV: "target-features"="+armv7-r,+dsp,+hwdiv,-thumb-mode"

// RUN: %clang_cc1 -triple thumbv7-linux-gnueabi -target-cpu cortex-m3 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-THUMB-DIV-M3
// CHECK-THUMB-DIV-M3: "target-features"="+armv7-m,+hwdiv,+thumb-mode"

// (The following test with no arch specified shouldn't happen; the driver
// rewrites triples.  Just make sure it does something sane.)
// RUN: %clang_cc1 -triple arm-linux-gnueabi -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARM-NOARCH-LINUX
// CHECK-ARM-NOARCH-LINUX: "target-features"="-thumb-mode"

// RUN: %clang_cc1 -triple armv4-linux-gnueabi -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARMV4-LINUX
// CHECK-ARMV4-LINUX: "target-features"="+armv4,-thumb-mode"

// RUN: %clang_cc1 -triple armv4t-linux-gnueabi -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARMV4T-LINUX
// CHECK-ARMV4T-LINUX: "target-features"="+armv4t,-thumb-mode"

// RUN: %clang_cc1 -triple armv5t-linux-gnueabi -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARMV5T-LINUX
// CHECK-ARMV5T-LINUX: "target-features"="+armv5t,-thumb-mode"

// RUN: %clang_cc1 -triple armv6-linux-gnueabi -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARMV6-LINUX
// CHECK-ARMV6-LINUX: "target-features"="+armv6,-thumb-mode"

// RUN: %clang_cc1 -triple armv6k-linux-gnueabi -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARMV6K-LINUX
// CHECK-ARMV6K-LINUX: "target-features"="+armv6k,-thumb-mode"

// RUN: %clang_cc1 -triple arm-linux-gnueabi -target-cpu mpcorenovfp -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARMV6K-MPCORE-LINUX
// CHECK-ARMV6K-MPCORE-LINUX: "target-features"="+armv6k,+dsp,-thumb-mode"

// RUN: %clang_cc1 -triple armv6t2-linux-gnueabi -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARMV6T2-LINUX
// CHECK-ARMV6T2-LINUX: "target-features"="+armv6t2,-thumb-mode"

// RUN: %clang_cc1 -triple thumbv6m-linux-gnueabi -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARMV6M-LINUX
// RUN: %clang_cc1 -triple thumb-linux-gnueabi -target-cpu cortex-m0 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARMV6M-LINUX 
// CHECK-ARMV6M-LINUX: "target-features"="+armv6-m,+thumb-mode"

// RUN: %clang_cc1 -triple thumbv7m-linux-gnueabi -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARMV7M-LINUX
// CHECK-ARMV7M-LINUX: "target-features"="+armv7-m,+thumb-mode"

// RUN: %clang_cc1 -triple thumb-linux-gnueabi -target-cpu cortex-m3 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARMV7M-M3-LINUX 
// CHECK-ARMV7M-M3-LINUX: "target-features"="+armv7-m,+hwdiv,+thumb-mode"

// RUN: %clang_cc1 -triple thumbv8m.base-linux-gnueabi -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARMV8M-LINUX
// CHECK-ARMV8M-LINUX: "target-features"="+armv8-m.base,+thumb-mode"

// RUN: %clang_cc1 -triple thumb-linux-gnueabi -target-cpu cortex-m23 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARMV8M-M23-LINUX 
// CHECK-ARMV8M-M23-LINUX: "target-features"="+armv8-m.base,+hwdiv,+thumb-mode"

// RUN: %clang_cc1 -triple thumb-linux-gnueabi -target-cpu cortex-m33 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARMV8M-MAIN-LINUX 
// CHECK-ARMV8M-MAIN-LINUX: "target-features"="+armv8-m.main,+d16,+dsp,+fp-armv8,+fp-only-sp,+hwdiv,+thumb-mode"

void foo() {}
