// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-apple-darwin9 | grep "global i32 10"

static int i;
int*j=&i;
static int i = 10;
