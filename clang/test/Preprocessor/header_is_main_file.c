// RUN: %clang_cc1 -x c-header -ffreestanding -Eonly -verify %s
// expected-no-diagnostics

#pragma once
#include_next "stdint.h"
#if !__has_include_next("stdint.h")
#error "__has_include_next failed"
#endif
