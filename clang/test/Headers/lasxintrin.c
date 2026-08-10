// RUN: %clang_cc1 %s -fsyntax-only -triple loongarch64 -target-feature +lasx
// RUN: %clang_cc1 %s -fsyntax-only -triple loongarch64 -target-feature +lasx -flax-vector-conversions=none
// RUN: %clang_cc1 %s -fsyntax-only -triple loongarch64 -target-feature +lasx -flax-vector-conversions=none -fno-signed-char

#include <lasxintrin.h>
