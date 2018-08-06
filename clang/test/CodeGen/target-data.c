// RUN: %clang_cc1 -triple i686-unknown-unknown -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=I686-UNKNOWN %s
// I686-UNKNOWN: target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"

// RUN: %clang_cc1 -triple i686-apple-darwin9 -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=I686-DARWIN %s
// I686-DARWIN: target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"

// RUN: %clang_cc1 -triple i686-unknown-win32 -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=I686-WIN32 %s
// I686-WIN32: target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"

// RUN: %clang_cc1 -triple i686-unknown-cygwin -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=I686-CYGWIN %s
// I686-CYGWIN: target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=X86_64 %s
// X86_64: target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

// RUN: %clang_cc1 -triple xcore-unknown-unknown -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=XCORE %s
// XCORE: target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:32-f64:32-a:0:32-n32"

// RUN: %clang_cc1 -triple sparc-sun-solaris -emit-llvm -o - %s | \
// RUN:     FileCheck %s --check-prefix=SPARC-V8
// SPARC-V8: target datalayout = "E-m:e-p:32:32-i64:64-f128:64-n32-S64"

// RUN: %clang_cc1 -triple sparcv9-sun-solaris -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefix=SPARC-V9
// SPARC-V9: target datalayout = "E-m:e-i64:64-n32:64-S128"

// RUN: %clang_cc1 -triple mipsel-linux-gnu -o - -emit-llvm %s |     \
// RUN: FileCheck %s -check-prefix=MIPS-32EL
// MIPS-32EL: target datalayout = "e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"

// RUN: %clang_cc1 -triple mips-linux-gnu -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=MIPS-32EB
// MIPS-32EB: target datalayout = "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"

// RUN: %clang_cc1 -triple mips64el-linux-gnu -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=MIPS-64EL
// MIPS-64EL: target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128"

// RUN: %clang_cc1 -triple mips64el-linux-gnu -o - -emit-llvm -target-abi n32 \
// RUN: %s | FileCheck %s -check-prefix=MIPS-64EL-N32
// MIPS-64EL-N32: target datalayout = "e-m:e-p:32:32-i8:8:32-i16:16:32-i64:64-n32:64-S128"

// RUN: %clang_cc1 -triple mips64-linux-gnu -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=MIPS-64EB
// MIPS-64EB: target datalayout = "E-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128"

// RUN: %clang_cc1 -triple mips64-linux-gnu -o - -emit-llvm %s -target-abi n32 \
// RUN: | FileCheck %s -check-prefix=MIPS-64EB-N32
// MIPS-64EB-N32: target datalayout = "E-m:e-p:32:32-i8:8:32-i16:16:32-i64:64-n32:64-S128"

// RUN: %clang_cc1 -triple powerpc64-lv2 -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=PS3
// PS3: target datalayout = "E-m:e-p:32:32-i64:64-n32:64"

// RUN: %clang_cc1 -triple i686-nacl -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=I686-NACL
// I686-NACL: target datalayout = "e-m:e-p:32:32-i64:64-n8:16:32-S128"

// RUN: %clang_cc1 -triple x86_64-nacl -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=X86_64-NACL
// X86_64-NACL: target datalayout = "e-m:e-p:32:32-i64:64-n8:16:32:64-S128"

// RUN: %clang_cc1 -triple arm-nacl -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=ARM-NACL
// ARM-NACL: target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S128"

// RUN: %clang_cc1 -triple mipsel-nacl -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=MIPS-NACL
// MIPS-NACL: target datalayout = "e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"

// RUN: %clang_cc1 -triple le32-nacl -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=LE32-NACL
// LE32-NACL: target datalayout = "e-p:32:32-i64:64"

// RUN: %clang_cc1 -triple wasm32-unknown-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=WEBASSEMBLY32
// WEBASSEMBLY32: target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"

// RUN: %clang_cc1 -triple wasm64-unknown-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=WEBASSEMBLY64
// WEBASSEMBLY64: target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"

// RUN: %clang_cc1 -triple lanai-unknown-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=LANAI
// LANAI: target datalayout = "E-m:e-p:32:32-i64:64-a:0:32-n32-S64"

// RUN: %clang_cc1 -triple powerpc-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=PPC
// PPC: target datalayout = "E-m:e-p:32:32-i64:64-n32"

// RUN: %clang_cc1 -triple powerpc64-freebsd -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=PPC64-FREEBSD
// PPC64-FREEBSD: target datalayout = "E-m:e-i64:64-n32:64"

// RUN: %clang_cc1 -triple powerpc64-linux -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=PPC64-LINUX
// PPC64-LINUX: target datalayout = "E-m:e-i64:64-n32:64"

// RUN: %clang_cc1 -triple powerpc64le-linux -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=PPC64LE-LINUX
// PPC64LE-LINUX: target datalayout = "e-m:e-i64:64-n32:64"

// RUN: %clang_cc1 -triple powerpc-darwin -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=PPC32-DARWIN
// PPC32-DARWIN: target datalayout = "E-m:o-p:32:32-f64:32:64-n32"

// RUN: %clang_cc1 -triple powerpc64-darwin -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=PPC64-DARWIN
// PPC64-DARWIN: target datalayout = "E-m:o-i64:64-n32:64"

// RUN: %clang_cc1 -triple nvptx-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=NVPTX
// NVPTX: target datalayout = "e-p:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"

// RUN: %clang_cc1 -triple nvptx64-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=NVPTX64
// NVPTX64: target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"

// RUN: %clang_cc1 -triple r600-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=R600
// R600: target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

// RUN: %clang_cc1 -triple r600-unknown -target-cpu cayman -o - -emit-llvm %s \
// RUN: | FileCheck %s -check-prefix=R600D
// R600D: target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

// RUN: %clang_cc1 -triple amdgcn-unknown -target-cpu hawaii -o - -emit-llvm %s \
// RUN: | FileCheck %s -check-prefix=R600SI
// R600SI: target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

// Test default -target-cpu
// RUN: %clang_cc1 -triple amdgcn-unknown -o - -emit-llvm %s \
// RUN: | FileCheck %s -check-prefix=R600SIDefault
// R600SIDefault: target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

// RUN: %clang_cc1 -triple arm64-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=AARCH64
// AARCH64: target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

// RUN: %clang_cc1 -triple thumb-unknown-gnueabi -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=THUMB
// THUMB: target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

// RUN: %clang_cc1 -triple arm-unknown-gnueabi -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=ARM
// ARM: target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

// RUN: %clang_cc1 -triple thumb-unknown -o - -emit-llvm -target-abi apcs-gnu \
// RUN: %s | FileCheck %s -check-prefix=THUMB-GNU
// THUMB-GNU: target datalayout = "e-m:e-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"

// RUN: %clang_cc1 -triple arm-unknown -o - -emit-llvm -target-abi apcs-gnu \
// RUN: %s | FileCheck %s -check-prefix=ARM-GNU
// ARM-GNU: target datalayout = "e-m:e-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"

// RUN: %clang_cc1 -triple hexagon-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=HEXAGON
// HEXAGON: target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"

// RUN: %clang_cc1 -triple s390x-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=SYSTEMZ
// SYSTEMZ: target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-a:8:16-n32:64"

// RUN: %clang_cc1 -triple s390x-unknown -target-cpu z13 -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=SYSTEMZ-VECTOR
// RUN: %clang_cc1 -triple s390x-unknown -target-cpu arch11 -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=SYSTEMZ-VECTOR
// RUN: %clang_cc1 -triple s390x-unknown -target-cpu z14 -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=SYSTEMZ-VECTOR
// RUN: %clang_cc1 -triple s390x-unknown -target-cpu arch12 -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=SYSTEMZ-VECTOR
// SYSTEMZ-VECTOR: target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"

// RUN: %clang_cc1 -triple msp430-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=MSP430
// MSP430: target datalayout = "e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16"

// RUN: %clang_cc1 -triple tce-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=TCE
// TCE: target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:32-v128:32:32-v256:32:32-v512:32:32-v1024:32:32-a0:0:32-n32"

// RUN: %clang_cc1 -triple tcele-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=TCELE
// TCELE: target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:32-v128:32:32-v256:32:32-v512:32:32-v1024:32:32-a0:0:32-n32"

// RUN: %clang_cc1 -triple spir-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=SPIR
// SPIR: target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"

// RUN: %clang_cc1 -triple spir64-unknown -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=SPIR64
// SPIR64: target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"

// RUN: %clang_cc1 -triple bpfel -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=BPFEL
// BPFEL: target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"

// RUN: %clang_cc1 -triple bpfeb -o - -emit-llvm %s | \
// RUN: FileCheck %s -check-prefix=BPFEB
// BPFEB: target datalayout = "E-m:e-p:64:64-i64:64-n32:64-S128"
