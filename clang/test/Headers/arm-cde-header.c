// RUN: %clang_cc1 -std=c89 -triple thumbv8.1m.main-none-none-eabi -target-feature +mve.fp -target-feature +cdecp0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c17 -triple thumbv8.1m.main-none-none-eabi -target-feature +mve -target-feature +cdecp0 -fsyntax-only %s
// RUN: %clang_cc1 -triple thumbv8.1m.main-none-none-eabi -target-feature +mve.fp -target-feature +cdecp0 -fsyntax-only %s
// RUN: %clang_cc1 -xc++ -std=c++98 -triple thumbv8.1m.main-none-none-eabi -target-feature +mve -target-feature +cdecp0 -fsyntax-only %s
// RUN: %clang_cc1 -xc++ -std=c++20 -triple thumbv8.1m.main-none-none-eabi -target-feature +mve.fp -target-feature +cdecp0 -fsyntax-only %s
// RUN: %clang_cc1 -xc++ -triple thumbv8.1m.main-none-none-eabi -target-feature +mve.fp -target-feature +cdecp0 -fsyntax-only %s

// REQUIRES: aarch64-registered-target || arm-registered-target

// Check that the headers don't conflict with each other

#include <arm_cde.h>
#include <arm_mve.h>
