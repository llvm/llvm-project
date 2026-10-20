// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

char *find(char *first, char *last, int value);

char *caller(char *first, char *last) { return find(first, last, 42); }

// A C function named like the std one lives outside std, so it carries no
// tag.
// CHECK: cir.func{{.*}} @find
// CHECK-NOT: func_identity
