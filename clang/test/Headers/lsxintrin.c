// RUN: %clang_cc1 %s -fsyntax-only -triple loongarch64 -target-feature +lsx
// RUN: %clang_cc1 %s -fsyntax-only -triple loongarch64 -target-feature +lsx -flax-vector-conversions=none
// RUN: %clang_cc1 %s -fsyntax-only -triple loongarch64 -target-feature +lsx -flax-vector-conversions=none -fno-signed-char

#include <lsxintrin.h>
