// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +pauth -flto=thin -emit-llvm -o - %s | FileCheck %s
// REQUIRES: aarch64-registered-target

// Check that +pauth target flag is enabled for global inline assembler.

// CHECK: module asm ".text"
// CHECK: module asm ".balign 16"
// CHECK: module asm ".globl foo"
// CHECK: module asm "pacib     x30, x27"
// CHECK: module asm "retab"
// CHECK: module asm ".previous"

asm (
    ".text" "\n"
    ".balign 16" "\n"
    ".globl foo\n"
    "pacib     x30, x27" "\n"
    "retab" "\n"
    ".previous" "\n"
);
