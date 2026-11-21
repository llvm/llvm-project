// RUN: %clang_cc1_no_such_shortcut -emit-llvm -o - %s | FileCheck %s

int zero(void) { return 0; }
