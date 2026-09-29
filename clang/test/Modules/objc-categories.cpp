// RUN: rm -rf %t.pcm

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -x c++ -std=c++11  -emit-module -fmodules -fmodule-name=cxx_library  %S/Inputs/module.modulemap -o %t.pcm
// RUN: llvm-bcanalyzer %t.pcm | FileCheck %s --check-prefix=CXX_LIBRARY
// CXX_LIBRARY: AST_BLOCK
// CXX_LIBRARY-NOT: OBJC_CATEGORIES
// CXX_LIBRARY-NOT: OBJC_CATEGORIES_MAP

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -x objective-c -emit-module -fmodules -fmodule-name=category_top  %S/Inputs/module.modulemap -o %t.pcm
// RUN: llvm-bcanalyzer %t.pcm | FileCheck %s --check-prefix=CATEGORY_TOP
// CATEGORY_TOP: AST_BLOCK
// CATEGORY_TOP: OBJC_CATEGORIES
// CATEGORY_TOP: OBJC_CATEGORIES_MAP

