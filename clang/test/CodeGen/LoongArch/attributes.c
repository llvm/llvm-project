// RUN: %clang_cc1 -emit-llvm -triple loongarch64 %s -o - | FileCheck %s

// CHECK: @normal ={{.*}} global i32 0, code_model "small"
int normal __attribute__((model("normal")));

// CHECK: @medium ={{.*}} global i32 0, code_model "medium"
int medium __attribute__((model("medium")));

// CHECK: @extreme ={{.*}} global i32 0, code_model "large"
int extreme __attribute__((model("extreme")));
