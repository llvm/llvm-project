// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.lp64.cir
// RUN: FileCheck --check-prefix=LP64 --input-file=%t.lp64.cir %s
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fclangir -emit-cir %s -o %t.ilp32.cir
// RUN: FileCheck --check-prefix=ILP32 --input-file=%t.ilp32.cir %s

// LP64: module{{.*}} attributes {{{.*}}cir.size_type_width = 64 : i32{{.*}}}
// ILP32: module{{.*}} attributes {{{.*}}cir.size_type_width = 32 : i32{{.*}}}

void f(void) {}
