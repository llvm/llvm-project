// RUN: %clang_cc1 -fxray-instrument -fxray-default-options='patch_premain=true,xray_mode=xray-basic' \
// RUN:     -std=c11 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

void justAFunction() {
}

// CHECK: !{{[0-9]+}} = !{i32 1, !"xray-default-opts", !"patch_premain=true,xray_mode=xray-basic"}
