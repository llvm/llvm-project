// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple wasm32-unknown-emscripten -fsanitize=address -emit-llvm -o - %s | FileCheck %s --check-prefix=EMSCRIPTEN

// CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @asan.module_ctor, ptr null }]
// EMSCRIPTEN: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 50, ptr @asan.module_ctor, ptr null }]
