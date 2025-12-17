// REQUIRES: riscv-registered-target
// RUN: rm -rf %t
// RUN: split-file %s %t


// Test precompiled header
// RUN: %clang_cc1 -triple riscv64-linux-gnu -target-feature +v -emit-pch -o %t/test_pch.pch %t/test_pch.h
// RUN: %clang_cc1 -triple riscv64-linux-gnu -target-feature +v -include-pch %t/test_pch.pch \
// RUN:   -fsyntax-only -verify %t/test_pch_src.c
//
// Test precompiled module(only available after C++20)
// RUN: %clang_cc1 -triple riscv64-linux-gnu -target-feature +v -std=c++20 -xc++-user-header -emit-header-unit -o %t/test_module1.pcm %t/test_module1.h
// RUN: %clang_cc1 -triple riscv64-linux-gnu -target-feature +v -std=c++20 -xc++-user-header -emit-header-unit -o %t/test_module2.pcm %t/test_module2.h
// RUN: %clang_cc1 -triple riscv64-linux-gnu -target-feature +v -std=c++20 -fmodule-file=%t/test_module1.pcm -fmodule-file=%t/test_module2.pcm \
// RUN:   -fsyntax-only %t/test_module_src.cpp

//--- test_pch.h
// expected-no-diagnostics
#include <riscv_vector.h>

//--- test_pch_src.c
// expected-no-diagnostics
vuint64m4_t v_add(vuint64m4_t a, vuint64m4_t b, size_t vl) {
    return __riscv_vadd_vv_u64m4(a, b, vl);
}

//--- test_module1.h
// expected-no-diagnostics
#include <riscv_vector.h>

//--- test_module2.h
// expected-no-diagnostics
// empty header

//--- test_module_src.cpp
// expected-no-diagnostics
import "test_module1.h";
import "test_module2.h";
vuint64m4_t v_add(vuint64m4_t a, vuint64m4_t b, size_t vl) {
    return __riscv_vadd_vv_u64m4(a, b, vl);
}
