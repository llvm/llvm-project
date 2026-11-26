// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -fsycl-is-device %s 2>&1 | FileCheck --check-prefixes=INVALID %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fsyntax-only -fsycl-is-device -verify=valid %s
// RUN: %clang_cc1 -fsyntax-only -fsycl-is-device -verify=valid %s

// These tests validate the target for SYCL device compilation

// INVALID: x86_64-unknown-unknown is not a supported SYCL device target

// valid-no-diagnostics

