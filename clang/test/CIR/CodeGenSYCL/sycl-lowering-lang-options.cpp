// RUN: %clang -fsycl -fsycl-device-only -fclangir -emit-cir %s -o - \
// RUN:   | FileCheck %s --check-prefix=DEVICE
// RUN: %clang --target=x86_64-unknown-linux-gnu -fsycl -fsycl-host-only \
// RUN:   -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=HOST

// The sycl_is_device field of #cir.lowering_options (see
// clang/test/CIR/CodeGen/lowering-lang-options.cpp) is serialized by CIRGen
// from LangOptions::SYCLIsDevice. This drives the full driver with
// -fsycl-device-only / -fsycl-host-only, rather than passing -fsycl-is-device
// / -fsycl-is-host to %clang_cc1 directly, so the check also covers the
// driver's translation of those user-facing flags into the -cc1 invocation
// that CIRGen runs under.

void foo() {}

// DEVICE: cir.lowering_options = #cir.lowering_options<
// DEVICE-SAME: sycl_is_device = true

// HOST: cir.lowering_options = #cir.lowering_options<
// HOST-SAME: sycl_is_device = false
