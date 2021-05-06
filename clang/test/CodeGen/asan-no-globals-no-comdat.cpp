// Test that on Linux asan constructor is placed in a comdat iff globals-gc is on.
// Even if there are no globals in the module.

// RUN: %clang_cc1 -fsanitize=address -fsanitize-address-globals-dead-stripping -emit-llvm -o - -triple x86_64-linux %s | FileCheck %s --check-prefix=WITHOUT-GC
// RUN: %clang_cc1 -fsanitize=address -fsanitize-address-globals-dead-stripping -fdata-sections -emit-llvm -o - -triple x86_64-linux %s | FileCheck %s --check-prefix=WITH-GC
// RUN: %clang_cc1 -fsanitize=address -fsanitize-address-globals-dead-stripping -fno-integrated-as -fdata-sections -emit-llvm -o - -triple x86_64-linux %s | FileCheck %s --check-prefix=WITHOUT-GC
// RUN: %clang_cc1 -fsanitize=address -fsanitize-address-globals-dead-stripping -fno-integrated-as -emit-llvm -o - -triple x86_64-linux %s | FileCheck %s --check-prefix=WITHOUT-GC
// RUN: %clang_cc1 -fsanitize=address -fdata-sections -emit-llvm -o - -triple x86_64-linux %s | FileCheck %s --check-prefix=WITHOUT-GC

// WITH-GC: define internal void @asan.module_ctor() #[[#]] comdat {
// WITHOUT-GC: define internal void @asan.module_ctor() #[[#]] {
