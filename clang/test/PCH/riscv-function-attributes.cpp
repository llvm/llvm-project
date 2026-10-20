// REQUIRES: riscv-registered-target

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -triple riscv64 -emit-pch -o %t/header.pch %t/header.h -x c++
// RUN: %clang_cc1 -triple riscv64 -include-pch %t/header.pch %t/main.cpp -fsyntax-only

//--- header.h
#pragma once

typedef void (*kernel_t)(unsigned long, float *) __riscv_in("xsfmm");

//--- main.cpp
typedef void (*kernel_t)(unsigned long, float *) __riscv_in("xsfmm");
