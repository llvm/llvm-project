// RUN: %clang_cc1 %s -emit-llvm -backend-option -time-passes -o - 2>&1 | FileCheck %s
// RUN: %clang_cc1 %s -emit-llvm -backend-option -time-passes -o - -triple spir-unknown-unknown 2>&1 | FileCheck %s
// CHECK: Pass execution timing report

