// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple x86_64-gnu-linux -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=LIN64
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple x86_64-windows-pc -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=WIN64
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple i386-gnu-linux -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=LIN32
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple i386-windows-pc -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=WIN32
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple nvptx64 -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=NVPTX64
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple nvptx -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=NVPTX
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple sparcv9 -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=SPARCV9
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple sparc -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=SPARC
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple mips64 -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=MIPS64
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple mips -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=MIPS
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple spir64 -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=SPIR64
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple spir -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=SPIR
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple hexagon -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=HEX
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple lanai -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=LANAI
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple r600 -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=R600
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple arc -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=ARC
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple xcore -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=XCORE
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple riscv64 -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=RISCV64
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple riscv32 -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=RISCV32
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple wasm64 -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=WASM
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple wasm32 -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=WASM
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple systemz -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=SYSTEMZ
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple ppc64 -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=PPC64
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple ppc -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=PPC32
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple aarch64 -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=AARCH64
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple aarch64 -target-abi darwinpcs -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=AARCH64DARWIN
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple arm64_32-apple-ios -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=AARCH64
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple arm64_32-apple-ios -target-abi darwinpcs -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=AARCH64DARWIN
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -triple arm -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=ARM

// Make sure 128 and 64 bit versions are passed like integers.
void ParamPassing(_BitInt(128) b, _BitInt(64) c) {}
// LIN64: define{{.*}} void @ParamPassing(i64 %{{.+}}, i64 %{{.+}}, i64 %{{.+}})
// WIN64: define dso_local void @ParamPassing(i128* %{{.+}}, i64 %{{.+}})
// LIN32: define{{.*}} void @ParamPassing(i128* %{{.+}}, i64 %{{.+}})
// WIN32: define dso_local void @ParamPassing(i128* %{{.+}}, i64 %{{.+}})
// NACL: define{{.*}} void @ParamPassing(i128* byval(i128) align 8 %{{.+}}, i64 %{{.+}})
// NVPTX64: define{{.*}} void @ParamPassing(i128 %{{.+}}, i64 %{{.+}})
// NVPTX: define{{.*}} void @ParamPassing(i128* byval(i128) align 8 %{{.+}}, i64 %{{.+}})
// SPARCV9: define{{.*}} void @ParamPassing(i128 %{{.+}}, i64 %{{.+}})
// SPARC: define{{.*}} void @ParamPassing(i128* byval(i128) align 8 %{{.+}}, i64 %{{.+}})
// MIPS64: define{{.*}} void @ParamPassing(i128 signext  %{{.+}}, i64 signext %{{.+}})
// MIPS: define{{.*}} void @ParamPassing(i128* byval(i128) align 8 %{{.+}}, i64 signext %{{.+}})
// SPIR64: define{{.*}} spir_func void @ParamPassing(i128* byval(i128) align 8 %{{.+}}, i64 %{{.+}})
// SPIR: define{{.*}} spir_func void @ParamPassing(i128* byval(i128) align 8 %{{.+}}, i64 %{{.+}})
// HEX: define{{.*}} void @ParamPassing(i128* byval(i128) align 8 %{{.+}}, i64 %{{.+}})
// LANAI: define{{.*}} void @ParamPassing(i128* byval(i128) align 4 %{{.+}}, i64 %{{.+}})
// R600: define{{.*}} void @ParamPassing(i128 addrspace(5)* byval(i128) align 8 %{{.+}}, i64 %{{.+}})
// ARC: define{{.*}} void @ParamPassing(i128* byval(i128) align 4 %{{.+}}, i64 inreg %{{.+}})
// XCORE: define{{.*}} void @ParamPassing(i128* byval(i128) align 4 %{{.+}}, i64 %{{.+}})
// RISCV64: define{{.*}} void @ParamPassing(i128 %{{.+}}, i64 %{{.+}})
// RISCV32: define{{.*}} void @ParamPassing(i128* %{{.+}}, i64 %{{.+}})
// WASM: define{{.*}} void @ParamPassing(i128 %{{.+}}, i64 %{{.+}})
// SYSTEMZ: define{{.*}} void @ParamPassing(i128* %{{.+}}, i64 %{{.+}})
// PPC64: define{{.*}} void @ParamPassing(i128 %{{.+}}, i64 %{{.+}})
// PPC32: define{{.*}} void @ParamPassing(i128* byval(i128) align 8 %{{.+}}, i64 %{{.+}})
// AARCH64: define{{.*}} void @ParamPassing(i128 %{{.+}}, i64 %{{.+}})
// AARCH64DARWIN: define{{.*}} void @ParamPassing(i128 %{{.+}}, i64 %{{.+}})
// ARM: define{{.*}} arm_aapcscc void @ParamPassing(i128* byval(i128) align 8 %{{.+}}, i64 %{{.+}})

void ParamPassing2(_BitInt(127) b, _BitInt(63) c) {}
// LIN64: define{{.*}} void @ParamPassing2(i64 %{{.+}}, i64 %{{.+}}, i64 %{{.+}})
// WIN64: define dso_local void @ParamPassing2(i127* %{{.+}}, i63 %{{.+}})
// LIN32: define{{.*}} void @ParamPassing2(i127* %{{.+}}, i63 %{{.+}})
// WIN32: define dso_local void @ParamPassing2(i127* %{{.+}}, i63 %{{.+}})
// NACL: define{{.*}} void @ParamPassing2(i127* byval(i127) align 8 %{{.+}}, i63 %{{.+}})
// NVPTX64: define{{.*}} void @ParamPassing2(i127 %{{.+}}, i63 %{{.+}})
// NVPTX: define{{.*}} void @ParamPassing2(i127* byval(i127) align 8 %{{.+}}, i63 %{{.+}})
// SPARCV9: define{{.*}} void @ParamPassing2(i127 %{{.+}}, i63 signext %{{.+}})
// SPARC: define{{.*}} void @ParamPassing2(i127* byval(i127) align 8 %{{.+}}, i63 %{{.+}})
// MIPS64: define{{.*}} void @ParamPassing2(i127 signext  %{{.+}}, i63 signext %{{.+}})
// MIPS: define{{.*}} void @ParamPassing2(i127* byval(i127) align 8 %{{.+}}, i63 signext %{{.+}})
// SPIR64: define{{.*}} spir_func void @ParamPassing2(i127* byval(i127) align 8 %{{.+}}, i63 %{{.+}})
// SPIR: define{{.*}} spir_func void @ParamPassing2(i127* byval(i127) align 8 %{{.+}}, i63 %{{.+}})
// HEX: define{{.*}} void @ParamPassing2(i127* byval(i127) align 8 %{{.+}}, i63 %{{.+}})
// LANAI: define{{.*}} void @ParamPassing2(i127* byval(i127) align 4 %{{.+}}, i63 %{{.+}})
// R600: define{{.*}} void @ParamPassing2(i127 addrspace(5)* byval(i127) align 8 %{{.+}}, i63 %{{.+}})
// ARC: define{{.*}} void @ParamPassing2(i127* byval(i127) align 4 %{{.+}}, i63 inreg %{{.+}})
// XCORE: define{{.*}} void @ParamPassing2(i127* byval(i127) align 4 %{{.+}}, i63 %{{.+}})
// RISCV64: define{{.*}} void @ParamPassing2(i127 %{{.+}}, i63 signext %{{.+}})
// RISCV32: define{{.*}} void @ParamPassing2(i127* %{{.+}}, i63 %{{.+}})
// WASM: define{{.*}} void @ParamPassing2(i127 %{{.+}}, i63 %{{.+}})
// SYSTEMZ: define{{.*}} void @ParamPassing2(i127* %{{.+}}, i63 signext %{{.+}})
// PPC64: define{{.*}} void @ParamPassing2(i127 %{{.+}}, i63 signext %{{.+}})
// PPC32: define{{.*}} void @ParamPassing2(i127* byval(i127) align 8 %{{.+}}, i63 %{{.+}})
// AARCH64: define{{.*}} void @ParamPassing2(i127 %{{.+}}, i63 %{{.+}})
// AARCH64DARWIN: define{{.*}} void @ParamPassing2(i127 %{{.+}}, i63 %{{.+}})
// ARM: define{{.*}} arm_aapcscc void @ParamPassing2(i127* byval(i127) align 8 %{{.+}}, i63 %{{.+}})

// Make sure we follow the signext rules for promotable integer types.
void ParamPassing3(_BitInt(15) a, _BitInt(31) b) {}
// LIN64: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// WIN64: define dso_local void @ParamPassing3(i15 %{{.+}}, i31 %{{.+}})
// LIN32: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// WIN32: define dso_local void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// NACL: define{{.*}} void @ParamPassing3(i15 %{{.+}}, i31 %{{.+}})
// NVPTX64: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// NVPTX: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// SPARCV9: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// SPARC: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// MIPS64: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// MIPS: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// SPIR64: define{{.*}} spir_func void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// SPIR: define{{.*}} spir_func void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// HEX: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// LANAI: define{{.*}} void @ParamPassing3(i15 inreg %{{.+}}, i31 inreg %{{.+}})
// R600: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// ARC: define{{.*}} void @ParamPassing3(i15 inreg signext %{{.+}}, i31 inreg signext %{{.+}})
// XCORE: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// RISCV64: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// RISCV32: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// WASM: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// SYSTEMZ: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// PPC64: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// PPC32: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// AARCH64: define{{.*}} void @ParamPassing3(i15 %{{.+}}, i31 %{{.+}})
// AARCH64DARWIN: define{{.*}} void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})
// ARM: define{{.*}} arm_aapcscc void @ParamPassing3(i15 signext %{{.+}}, i31 signext %{{.+}})

#if __BITINT_MAXWIDTH__ > 128
// When supported, bit-precise types that are >128 are passed indirectly. Note,
// FileCheck doesn't pay attention to the preprocessor, so all of these tests
// are negated. This will give an error when a target does support larger
// _BitInt widths to alert us to enable the test.
void ParamPassing4(_BitInt(129) a) {}
// LIN64-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
// WIN64-NOT: define dso_local void @ParamPassing4(i129* %{{.+}})
// LIN32-NOT: define{{.*}} void @ParamPassing4(i129* %{{.+}})
// WIN32-NOT: define dso_local void @ParamPassing4(i129* %{{.+}})
// NACL-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
// NVPTX64-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
// NVPTX-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
// SPARCV9-NOT: define{{.*}} void @ParamPassing4(i129* %{{.+}})
// SPARC-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
// MIPS64-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
// MIPS-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
// SPIR64-NOT: define{{.*}} spir_func void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
// SPIR-NOT: define{{.*}} spir_func void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
// HEX-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
// LANAI-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 4 %{{.+}})
// R600-NOT: define{{.*}} void @ParamPassing4(i129 addrspace(5)* byval(i129) align 8 %{{.+}})
// ARC-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 4 %{{.+}})
// XCORE-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 4 %{{.+}})
// RISCV64-NOT: define{{.*}} void @ParamPassing4(i129* %{{.+}})
// RISCV32-NOT: define{{.*}} void @ParamPassing4(i129* %{{.+}})
// WASM-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
// SYSTEMZ-NOT: define{{.*}} void @ParamPassing4(i129* %{{.+}})
// PPC64-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
// PPC32-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
// AARCH64-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
// AARCH64DARWIN-NOT: define{{.*}} void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
// ARM-NOT: define{{.*}} arm_aapcscc void @ParamPassing4(i129* byval(i129) align 8 %{{.+}})
#endif

_BitInt(63) ReturnPassing(void){}
// LIN64: define{{.*}} i64 @ReturnPassing(
// WIN64: define dso_local i63 @ReturnPassing(
// LIN32: define{{.*}} i63 @ReturnPassing(
// WIN32: define dso_local i63 @ReturnPassing(
// NACL: define{{.*}} i63 @ReturnPassing(
// NVPTX64: define{{.*}} i63 @ReturnPassing(
// NVPTX: define{{.*}} i63 @ReturnPassing(
// SPARCV9: define{{.*}} signext i63 @ReturnPassing(
// SPARC: define{{.*}} i63 @ReturnPassing(
// MIPS64: define{{.*}} i63 @ReturnPassing(
// MIPS: define{{.*}} i63 @ReturnPassing(
// SPIR64: define{{.*}} spir_func i63 @ReturnPassing(
// SPIR: define{{.*}} spir_func i63 @ReturnPassing(
// HEX: define{{.*}} i63 @ReturnPassing(
// LANAI: define{{.*}} i63 @ReturnPassing(
// R600: define{{.*}} i63 @ReturnPassing(
// ARC: define{{.*}} i63 @ReturnPassing(
// XCORE: define{{.*}} i63 @ReturnPassing(
// RISCV64: define{{.*}} signext i63 @ReturnPassing(
// RISCV32: define{{.*}} i63 @ReturnPassing(
// WASM: define{{.*}} i63 @ReturnPassing(
// SYSTEMZ: define{{.*}} signext i63 @ReturnPassing(
// PPC64: define{{.*}} signext i63 @ReturnPassing(
// PPC32: define{{.*}} i63 @ReturnPassing(
// AARCH64: define{{.*}} i63 @ReturnPassing(
// AARCH64DARWIN: define{{.*}} i63 @ReturnPassing(
// ARM: define{{.*}} arm_aapcscc i63 @ReturnPassing(

_BitInt(64) ReturnPassing2(void){}
// LIN64: define{{.*}} i64 @ReturnPassing2(
// WIN64: define dso_local i64 @ReturnPassing2(
// LIN32: define{{.*}} i64 @ReturnPassing2(
// WIN32: define dso_local i64 @ReturnPassing2(
// NACL: define{{.*}} i64 @ReturnPassing2(
// NVPTX64: define{{.*}} i64 @ReturnPassing2(
// NVPTX: define{{.*}} i64 @ReturnPassing2(
// SPARCV9: define{{.*}} i64 @ReturnPassing2(
// SPARC: define{{.*}} i64 @ReturnPassing2(
// MIPS64: define{{.*}} i64 @ReturnPassing2(
// MIPS: define{{.*}} i64 @ReturnPassing2(
// SPIR64: define{{.*}} spir_func i64 @ReturnPassing2(
// SPIR: define{{.*}} spir_func i64 @ReturnPassing2(
// HEX: define{{.*}} i64 @ReturnPassing2(
// LANAI: define{{.*}} i64 @ReturnPassing2(
// R600: define{{.*}} i64 @ReturnPassing2(
// ARC: define{{.*}} i64 @ReturnPassing2(
// XCORE: define{{.*}} i64 @ReturnPassing2(
// RISCV64: define{{.*}} i64 @ReturnPassing2(
// RISCV32: define{{.*}} i64 @ReturnPassing2(
// WASM: define{{.*}} i64 @ReturnPassing2(
// SYSTEMZ: define{{.*}} i64 @ReturnPassing2(
// PPC64: define{{.*}} i64 @ReturnPassing2(
// PPC32: define{{.*}} i64 @ReturnPassing2(
// AARCH64: define{{.*}} i64 @ReturnPassing2(
// AARCH64DARWIN: define{{.*}} i64 @ReturnPassing2(
// ARM: define{{.*}} arm_aapcscc i64 @ReturnPassing2(

_BitInt(127) ReturnPassing3(void){}
// LIN64: define{{.*}} { i64, i64 } @ReturnPassing3(
// WIN64: define dso_local void @ReturnPassing3(i127* noalias sret
// LIN32: define{{.*}} void @ReturnPassing3(i127* noalias sret
// WIN32: define dso_local void @ReturnPassing3(i127* noalias sret
// NACL: define{{.*}} void @ReturnPassing3(i127* noalias sret
// NVPTX/64 makes the intentional choice to put all return values direct, even
// large structures, so we do the same here.
// NVPTX64: define{{.*}} i127 @ReturnPassing3(
// NVPTX: define{{.*}} i127 @ReturnPassing3(
// SPARCV9: define{{.*}} i127 @ReturnPassing3(
// SPARC: define{{.*}} void @ReturnPassing3(i127* noalias sret
// MIPS64: define{{.*}} i127 @ReturnPassing3(
// MIPS: define{{.*}} void @ReturnPassing3(i127* noalias sret
// SPIR64: define{{.*}} spir_func void @ReturnPassing3(i127* noalias sret
// SPIR: define{{.*}} spir_func void @ReturnPassing3(i127* noalias sret
// HEX: define{{.*}} void @ReturnPassing3(i127* noalias sret
// LANAI: define{{.*}} void @ReturnPassing3(i127* noalias sret
// R600: define{{.*}} void @ReturnPassing3(i127 addrspace(5)* noalias sret
// ARC: define{{.*}} void @ReturnPassing3(i127* noalias sret
// XCORE: define{{.*}} void @ReturnPassing3(i127* noalias sret
// RISCV64: define{{.*}} i127 @ReturnPassing3(
// RISCV32: define{{.*}} void @ReturnPassing3(i127* noalias sret
// WASM: define{{.*}} i127 @ReturnPassing3(
// SYSTEMZ: define{{.*}} void @ReturnPassing3(i127* noalias sret
// PPC64: define{{.*}} i127 @ReturnPassing3(
// PPC32: define{{.*}} void @ReturnPassing3(i127* noalias sret
// AARCH64: define{{.*}} i127 @ReturnPassing3(
// AARCH64DARWIN: define{{.*}} i127 @ReturnPassing3(
// ARM: define{{.*}} arm_aapcscc void @ReturnPassing3(i127* noalias sret

_BitInt(128) ReturnPassing4(void){}
// LIN64: define{{.*}} { i64, i64 } @ReturnPassing4(
// WIN64: define dso_local void @ReturnPassing4(i128* noalias sret
// LIN32: define{{.*}} void @ReturnPassing4(i128* noalias sret
// WIN32: define dso_local void @ReturnPassing4(i128* noalias sret
// NACL: define{{.*}} void @ReturnPassing4(i128* noalias sret
// NVPTX64: define{{.*}} i128 @ReturnPassing4(
// NVPTX: define{{.*}} i128 @ReturnPassing4(
// SPARCV9: define{{.*}} i128 @ReturnPassing4(
// SPARC: define{{.*}} void @ReturnPassing4(i128* noalias sret
// MIPS64: define{{.*}} i128 @ReturnPassing4(
// MIPS: define{{.*}} void @ReturnPassing4(i128* noalias sret
// SPIR64: define{{.*}} spir_func void @ReturnPassing4(i128* noalias sret
// SPIR: define{{.*}} spir_func void @ReturnPassing4(i128* noalias sret
// HEX: define{{.*}} void @ReturnPassing4(i128* noalias sret
// LANAI: define{{.*}} void @ReturnPassing4(i128* noalias sret
// R600: define{{.*}} void @ReturnPassing4(i128 addrspace(5)* noalias sret
// ARC: define{{.*}} void @ReturnPassing4(i128* noalias sret
// XCORE: define{{.*}} void @ReturnPassing4(i128* noalias sret
// RISCV64: define{{.*}} i128 @ReturnPassing4(
// RISCV32: define{{.*}} void @ReturnPassing4(i128* noalias sret
// WASM: define{{.*}} i128 @ReturnPassing4(
// SYSTEMZ: define{{.*}} void @ReturnPassing4(i128* noalias sret
// PPC64: define{{.*}} i128 @ReturnPassing4(
// PPC32: define{{.*}} void @ReturnPassing4(i128* noalias sret
// AARCH64: define{{.*}} i128 @ReturnPassing4(
// AARCH64DARWIN: define{{.*}} i128 @ReturnPassing4(
// ARM: define{{.*}} arm_aapcscc void @ReturnPassing4(i128* noalias sret

#if __BITINT_MAXWIDTH__ > 128
_BitInt(129) ReturnPassing5(void){}
// LIN64-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// WIN64-NOT: define dso_local void @ReturnPassing5(i129* noalias sret
// LIN32-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// WIN32-NOT: define dso_local void @ReturnPassing5(i129* noalias sret
// NACL-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// NVPTX64-NOT: define{{.*}} i129 @ReturnPassing5(
// NVPTX-NOT: define{{.*}} i129 @ReturnPassing5(
// SPARCV9-NOT: define{{.*}} i129 @ReturnPassing5(
// SPARC-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// MIPS64-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// MIPS-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// SPIR64-NOT: define{{.*}} spir_func void @ReturnPassing5(i129* noalias sret
// SPIR-NOT: define{{.*}} spir_func void @ReturnPassing5(i129* noalias sret
// HEX-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// LANAI-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// R600-NOT: define{{.*}} void @ReturnPassing5(i129 addrspace(5)* noalias sret
// ARC-NOT: define{{.*}} void @ReturnPassing5(i129* inreg noalias sret
// XCORE-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// RISCV64-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// RISCV32-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// WASM-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// SYSTEMZ-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// PPC64-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// PPC32-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// AARCH64-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// AARCH64DARWIN-NOT: define{{.*}} void @ReturnPassing5(i129* noalias sret
// ARM-NOT: define{{.*}} arm_aapcscc void @ReturnPassing5(i129* noalias sret

// SparcV9 is odd in that it has a return-size limit of 256, not 128 or 64
// like other platforms, so test to make sure this behavior will still work.
_BitInt(256) ReturnPassing6(void) {}
// SPARCV9-NOT: define{{.*}} i256 @ReturnPassing6(
_BitInt(257) ReturnPassing7(void) {}
// SPARCV9-NOT: define{{.*}} void @ReturnPassing7(i257* noalias sret
#endif
