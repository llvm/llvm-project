/// Check the sse/sse2 device features derived from the host target.

// RUN: %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=SSE2 %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=SSE2 %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple arm64ec-pc-windows-msvc \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=SSE2 %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=SSE2 %s
// RUN: %clang_cc1 -triple spirv32-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=SSE2 %s
/// Windows ARM64 without EC is not an x86 host and does not predefine _M_X64.
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple aarch64-pc-windows-msvc \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=NO-SSE2 %s

/// 32-bit Windows hosts predefine _M_IX86 rather than _M_X64.
// RUN: %clang_cc1 -triple spirv32-unknown-unknown -aux-triple i386-pc-windows-msvc \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=NO-SSE2 %s

/// Non-MSVC x86_64 hosts do not predefine _M_X64, whether or not they are
/// Windows hosts.
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=NO-SSE2 %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-gnu \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=NO-SSE2 %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-uefi \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=NO-SSE2 %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=NO-SSE2 %s

[[clang::sycl_external]] void test() {}

// SSE2: define {{.*}}spir_func void @_Z4testv() #[[ATTR:[0-9]+]]
// SSE2: attributes #[[ATTR]] = {{{.*}}"target-features"="+sse,+sse2"

// NO-SSE2-NOT: "target-features"
