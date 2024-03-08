// Test the output from -module-file-info about C++20 Modules
// can reflect macros definitions correctly.
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/foo.h -o %t/foo.pcm
// RUN: %clang_cc1 -module-file-info %t/foo.pcm | FileCheck %t/foo.h
//
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/include_foo.h -o %t/include_foo.pcm
// RUN: %clang_cc1 -module-file-info %t/include_foo.pcm | FileCheck %t/include_foo.h

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header -fmodule-file=%t/foo.pcm \
// RUN:     %t/import_foo.h -o %t/import_foo.pcm
// RUN: %clang_cc1 -module-file-info %t/import_foo.pcm | FileCheck %t/import_foo.h
//
// RUN: %clang_cc1 -std=c++20 %t/named_module.cppm -emit-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -module-file-info %t/M.pcm | FileCheck %t/named_module.cppm

// RUN: %clang_cc1 -std=c++20 %t/named_module.cppm -emit-reduced-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -module-file-info %t/M.pcm | FileCheck %t/named_module.cppm

//--- foo.h
#pragma once
#define FOO
#define CONSTANT 43
#define FUNC_Macro(X) (X+1)
#define TO_BE_UNDEF
#undef TO_BE_UNDEF

#ifndef FOO
#define CONDITIONAL_DEF
#endif

#define REDEFINE
#define REDEFINE

// CHECK: Macro Definitions:
// CHECK-DAG: REDEFINE
// CHECK-DAG: FUNC_Macro
// CHECK-DAG: CONSTANT
// CHECK-DAG: FOO
// CHECK-NEXT: ===

//--- include_foo.h
#include "foo.h"
#undef REDEFINE
// CHECK: Macro Definitions:
// CHECK-DAG: CONSTANT
// CHECK-DAG: FUNC_Macro
// CHECK-DAG: FOO
// CHECK-NEXT: ===

//--- import_foo.h
import "foo.h";
#undef REDEFINE
// CHECK: Macro Definitions:
// CHECK-DAG: CONSTANT
// CHECK-DAG: FUNC_Macro
// CHECK-DAG: FOO
// CHECK-NEXT: ===

//--- named_module.cppm
module;
#include "foo.h"
export module M;
#define M_Module 43
// CHECK-NOT: Macro Definitions:
