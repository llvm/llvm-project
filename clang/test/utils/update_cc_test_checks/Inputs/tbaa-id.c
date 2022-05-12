// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

void foo(unsigned **ptr) {
  *ptr = 0;
}
