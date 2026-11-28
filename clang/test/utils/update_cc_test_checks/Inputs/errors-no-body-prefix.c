// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefixes=C,LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - -x c++ %s | FileCheck %s --check-prefix=CPP

int zero(void) { return 0; }
