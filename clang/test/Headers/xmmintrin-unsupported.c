// RUN: not %clang_cc1 %s -triple aarch64-eabi -fsyntax-only 2>&1 | FileCheck %s
//
// REQUIRES: x86-registered-target
// CHECK: This header is only meant to be used on x86 and x64 architecture
#include <xmmintrin.h>
