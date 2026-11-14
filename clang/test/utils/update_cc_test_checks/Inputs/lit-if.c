// Check that the commands of a '%if' RUN line are run.
// RUN: %if x86-registered-target %{ %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s %}

void f(void) {}
