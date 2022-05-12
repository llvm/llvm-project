// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s
__attribute__((error("oh no"))) void foo(void);

void bar(void) {
  foo();
}
