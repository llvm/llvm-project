// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple wasm32-unknown-emscripten -fsanitize=address -emit-llvm -o - %s | FileCheck %s --check-prefix=EMSCRIPTEN

// CHECK: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 1, void ()* @asan.module_ctor, i8* null }]
// EMSCRIPTEN: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 50, void ()* @asan.module_ctor, i8* null }]
