// RUN: %clang_cc1 -E %s | FileCheck %s

#pragma clang __debug parser_crash
#pragma clang __debug dump Test

// CHECK: #pragma clang __debug parser_crash
// FIXME: The dump parameter is dropped.
// CHECK: #pragma clang __debug dump{{$}}
