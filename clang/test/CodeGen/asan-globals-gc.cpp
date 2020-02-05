// RUN: %clang_cc1 -fsanitize=address -fsanitize-address-globals-dead-stripping -emit-llvm -o - -triple x86_64-linux %s | FileCheck %s --check-prefix=WITHOUT-GC
// RUN: %clang_cc1 -fsanitize=address -fsanitize-address-globals-dead-stripping -fdata-sections -emit-llvm -o - -triple x86_64-linux %s | FileCheck %s --check-prefix=WITH-GC
// RUN: %clang_cc1 -fsanitize=address -fsanitize-address-globals-dead-stripping -fno-integrated-as -fdata-sections -emit-llvm -o - -triple x86_64-linux %s | FileCheck %s --check-prefix=WITHOUT-GC
// RUN: %clang_cc1 -fsanitize=address -fsanitize-address-globals-dead-stripping -fno-integrated-as -emit-llvm -o - -triple x86_64-linux %s | FileCheck %s --check-prefix=WITHOUT-GC
// RUN: %clang_cc1 -fsanitize=address -fdata-sections -emit-llvm -o - -triple x86_64-linux %s | FileCheck %s --check-prefix=WITHOUT-GC

// RUN: %clang_cc1 -fsanitize=address -fsanitize-address-globals-dead-stripping -emit-llvm -o - -triple x86_64-windows-msvc %s | FileCheck %s --check-prefix=WITH-GC
// RUN: %clang_cc1 -fsanitize=address -fsanitize-address-globals-dead-stripping -fdata-sections -emit-llvm -o - -triple x86_64-windows-msvc %s | FileCheck %s --check-prefix=WITH-GC
// RUN: %clang_cc1 -fsanitize=address -fdata-sections -emit-llvm -o - -triple x86_64-windows-msvc %s | FileCheck %s --check-prefix=WITHOUT-GC

// RUN: %clang_cc1 -fsanitize=address -fsanitize-address-globals-dead-stripping -emit-llvm -o - -triple x86_64-apple-macosx11 %s | FileCheck %s --check-prefix=WITH-GC
// RUN: %clang_cc1 -fsanitize=address -fsanitize-address-globals-dead-stripping -fdata-sections -emit-llvm -o - -triple x86_64-apple-macosx11 %s | FileCheck %s --check-prefix=WITH-GC
// RUN: %clang_cc1 -fsanitize=address -fdata-sections -emit-llvm -o - -triple x86_64-apple-macosx11 %s | FileCheck %s --check-prefix=WITHOUT-GC

int global;

// WITH-GC-NOT: call void @__asan_register_globals
// WITHOUT-GC: call void @__asan_register_globals
