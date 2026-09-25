// SYCL enables relocatable device code by default. SYCL does not externalize
// file-scope statics, so -fgpu-rdc must not change mangled names on either the
// host or the device side.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host -fgpu-rdc \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host -fgpu-rdc \
// RUN:   -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host -fgpu-rdc \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device -fgpu-rdc \
// RUN:   -fclangir -emit-cir %s -o %t-dev.cir
// RUN: FileCheck --check-prefix=CIR-DEV --input-file=%t-dev.cir %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device -fgpu-rdc \
// RUN:   -fclangir -emit-llvm %s -o %t-dev-cir.ll
// RUN: FileCheck --check-prefix=LLVM-DEV --input-file=%t-dev-cir.ll %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device -fgpu-rdc \
// RUN:   -emit-llvm %s -o %t-dev.ll
// RUN: FileCheck --check-prefix=LLVM-DEV --input-file=%t-dev.ll %s

template <typename KN, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KN, typename KT>
[[clang::sycl_kernel_entry_point(KN)]] void kernel_entry(KT k) { k(); }

struct KN;

static int hostStatic;

template <typename T> static T getValue() { return T(1); }

int main() {
  hostStatic = 1;
  kernel_entry<KN>([] { (void)getValue<int>(); });
}

// CIR: cir.global "private" internal dso_local @_ZL10hostStatic
// CIR: cir.func {{.*}}@main
// LLVM: @_ZL10hostStatic = internal global i32 0
// LLVM: define {{.*}}@main

// CIR-DEV: cir.func {{.*}}@_ZTS2KN
// CIR-DEV: cir.func {{.*}}@_ZL8getValueIiET_v
// LLVM-DEV: define {{.*}}spir_kernel void @_ZTS2KN
// LLVM-DEV: define internal spir_func {{.*}}@_ZL8getValueIiET_v
