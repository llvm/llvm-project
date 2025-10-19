// RUN: %clang_cc1 -triple armv7-eabi -target-cpu cortex-a15 -fsyntax-only -ffreestanding %s
// RUN: %clang_cc1 -triple aarch64 -target-cpu cortex-a53 -fsyntax-only -ffreestanding %s
// RUN: %clang_cc1 -triple thumbv7-windows -target-cpu cortex-a53 -fsyntax-only -ffreestanding %s
// RUN: %clang_cc1 -x c++ -triple armv7-eabi -target-cpu cortex-a15 -fsyntax-only -ffreestanding %s
// RUN: %clang_cc1 -x c++ -triple aarch64 -target-cpu cortex-a57 -fsyntax-only -ffreestanding %s
// RUN: %clang_cc1 -x c++ -triple thumbv7-windows -target-cpu cortex-a15 -fsyntax-only -ffreestanding %s
// RUN: %clang_cc1 -x c++ -triple thumbv7-windows -target-cpu cortex-a15 -fsyntax-only -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=19.11 %s
// RUN: %clang_cc1 -x c++ -triple aarch64-windows -target-cpu cortex-a53 -fsyntax-only -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=19.11 %s
// RUN: %clang_cc1 -x c++ -triple arm64-apple-ios -target-cpu apple-a7 -fsyntax-only -ffreestanding -fms-extensions %s
// RUN: %clang_cc1 -x c++ -triple arm64ec-windows -target-cpu cortex-a53 -fsyntax-only -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=19.11 %s
// expected-no-diagnostics

#include "system_reserved_names.h"

#include <arm_acle.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
void f(void) { __nop(); __dmb(0); __wfi(); }
