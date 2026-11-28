// Check that '%if's are unconditionally unwrapped, discarding %else branches.
// RUN: %if x86-registered-target %{ %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s %}
// RUN: %clang_cc1 %if x86-registered-target %{ -triple x86_64-unknown-linux-gnu %} -emit-llvm %s -o - | FileCheck %s
// RUN: %if x86-registered-target %{ %clang_cc1 %if x86-registered-target %{ -triple x86_64-unknown-linux-gnu %} -emit-llvm %s -o - | FileCheck %s %}
// RUN: %if x86-registered-target %{ %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s %} %else %{ %clang_cc1 -emit-llvm %s -o - | FileCheck %s --check-prefix=ELSE %}

void f(void) {}
