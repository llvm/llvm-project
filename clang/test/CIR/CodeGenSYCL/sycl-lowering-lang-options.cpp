// RUN: %clang_cc1 -std=c++17 -triple spirv64-unknown-unknown -fsycl-is-device \
// RUN:   -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=DEVICE
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=HOST

void foo() {}

// DEVICE: cir.lowering_options = #cir.lowering_options<
// DEVICE-SAME: sycl_is_device = true

// HOST: cir.lowering_options = #cir.lowering_options<
// HOST-SAME: sycl_is_device = false
