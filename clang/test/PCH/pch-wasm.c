// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-pch -fmodule-format=obj %S/pchpch1.h -o - | llvm-objdump --section-headers - | FileCheck %s

// Ensure that clangast section should be emitted in a section for wasm object file

// CHECK: file format wasm
// CHECK: __clangast   {{[0-9a-f]+}} {{[0-9a-f]+}}
