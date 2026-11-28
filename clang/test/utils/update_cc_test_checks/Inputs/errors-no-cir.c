// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -o - %s | FileCheck %s

int zero(void) { return 0; }
