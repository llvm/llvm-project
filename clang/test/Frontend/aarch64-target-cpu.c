// Ensure we support the various CPU names.
//
// RUN: %clang_cc1 -triple aarch64-unknown-unknown -target-cpu cortex-a35 -verify %s
// RUN: %clang_cc1 -triple aarch64-unknown-unknown -target-cpu cortex-a53 -verify %s
// RUN: %clang_cc1 -triple aarch64-unknown-unknown -target-cpu cortex-a57 -verify %s
// RUN: %clang_cc1 -triple aarch64-unknown-unknown -target-cpu cortex-a72 -verify %s
// RUN: %clang_cc1 -triple aarch64-unknown-unknown -target-cpu cortex-a73 -verify %s
// RUN: %clang_cc1 -triple aarch64-unknown-unknown -target-cpu cyclone -verify %s
// RUN: %clang_cc1 -triple aarch64-unknown-unknown -target-cpu exynos-m1 -verify %s
// RUN: %clang_cc1 -triple aarch64-unknown-unknown -target-cpu generic -verify %s
// RUN: %clang_cc1 -triple aarch64-unknown-unknown -target-cpu kryo -verify %s
// RUN: %clang_cc1 -triple aarch64-unknown-unknown -target-cpu thunderx2t99 -verify %s
//
// expected-no-diagnostics
