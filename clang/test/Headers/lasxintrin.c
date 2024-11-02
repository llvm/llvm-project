// RUN: %clang_cc1 %s -fsyntax-only -triple loongarch64 -target-feature +lasx
// RUN: not %clang_cc1 %s -fsyntax-only -triple loongarch64 -target-feature +lasx -flax-vector-conversions=none
// RUN: not %clang_cc1 %s -fsyntax-only -triple loongarch64 -target-feature +lasx -flax-vector-conversions=none -fno-signed-char
// FIXME: "not" should be removed once we fix GH#110834.

#include <lasxintrin.h>
