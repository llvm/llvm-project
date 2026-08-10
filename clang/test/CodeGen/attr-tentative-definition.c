// RUN: %clang_cc1 -emit-llvm -triple x86_64-linux-unknown < %s | FileCheck %s

char arr[10];
char arr[10] __attribute__((section("datadata")));
char arr[10] __attribute__((aligned(16)));

// CHECK: @arr ={{.*}}section "datadata", align 16
