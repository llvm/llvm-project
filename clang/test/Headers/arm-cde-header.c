// RUN: %clang_cc1 -std=c89 -triple thumbv8.1m.main-arm-none-eabi -fallow-half-arguments-and-returns -target-feature +mve.fp -target-feature +cdecp0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c99 -triple thumbv8.1m.main-arm-none-eabi -fallow-half-arguments-and-returns -target-feature +mve.fp -target-feature +cdecp0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11 -triple thumbv8.1m.main-arm-none-eabi -fallow-half-arguments-and-returns -target-feature +mve.fp -target-feature +cdecp0 -fsyntax-only %s
// RUN: %clang_cc1 -triple thumbv8.1m.main-arm-none-eabi -fallow-half-arguments-and-returns -target-feature +mve.fp -target-feature +cdecp0 -fsyntax-only %s

// Check that the headers don't conflict with each other
#include <arm_cde.h>
#include <arm_mve.h>
