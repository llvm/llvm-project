// RUN: %clang_cc1 -triple x86_64-pc-win32 \
// RUN:     -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:     -ffreestanding -fsyntax-only -Werror -Wsystem-headers \
// RUN:     -isystem %S/Inputs/include %s

#include <immintrin.h>

#pragma clang attribute push(__attribute__((target("avx"))), apply_to=function)
#include <intrin.h>
#pragma clang attribute pop
