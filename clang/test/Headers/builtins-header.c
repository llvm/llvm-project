// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -ffreestanding -emit-llvm -o - %s | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple powerpc64-unknown-unknown -ffreestanding -emit-llvm -o - %s | FileCheck %s %}
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -ffreestanding -emit-llvm -o - %s | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple powerpc64le-unknown-unknown -ffreestanding -emit-llvm -o - %s | FileCheck %s %}

#include <builtins.h>

// Verify that we can include <builtins.h>

// CHECK: target triple = "powerpc64
