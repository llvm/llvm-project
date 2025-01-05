// Test that the DSO-local runtime library has been linked if -fxray-shared is passed.
//
// RUN: %clangxx -fxray-instrument -fxray-shared -fPIC %s -shared -o %t.so
// RUN: llvm-nm %t.so | FileCheck %s --check-prefix ENABLED

// RUN: %clangxx -fxray-instrument %s -shared -o %t.so
// RUN: llvm-nm %t.so | FileCheck %s --check-prefix DISABLED
//
// REQUIRES: target={{(aarch64|x86_64)-.*}}

[[clang::xray_always_instrument]] int always_instrumented() { return 42; }

// ENABLED: __start_xray_instr_map
// DISABLED-NOT: __start_xray_instr_map
