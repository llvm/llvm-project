// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=aarch64-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=arm-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=i386-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=mips-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=mips64-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=msp430-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=powerpc64-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=powerpc64-none-netbsd
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=powerpc-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=s390x-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=sparc-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=tce-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=x86_64-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=x86_64-pc-linux-gnu
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=i386-mingw32
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only -triple=xcore-none-none
// RUN: %clang_cc1 %s -ffreestanding -std=c++1z -fsyntax-only

#include <stdint.h>
#include <stddef.h>

static_assert(__is_same(__typeof__(INTPTR_MIN), intptr_t));
static_assert(__is_same(__typeof__(INTPTR_MAX), intptr_t));
static_assert(__is_same(__typeof__(UINTPTR_MAX), uintptr_t));
static_assert(__is_same(__typeof__(PTRDIFF_MIN), ptrdiff_t));
static_assert(__is_same(__typeof__(PTRDIFF_MAX), ptrdiff_t));
static_assert(__is_same(__typeof__(SIZE_MAX), size_t));
static_assert(__is_same(__typeof__(INTMAX_MIN), intmax_t));
static_assert(__is_same(__typeof__(INTMAX_MAX), intmax_t));
static_assert(__is_same(__typeof__(UINTMAX_MAX), uintmax_t));
static_assert(__is_same(__typeof__(INTMAX_C(5)), intmax_t));
static_assert(__is_same(__typeof__(UINTMAX_C(5)), uintmax_t));
