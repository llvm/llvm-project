// RUN: %clang_cc1 -triple=x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s

struct bork {
  struct bork *next_local;
  char * query;
};
int offset =  (char *) &(((struct bork *) 0x10)->query) - (char *) 0x10;
// CHECK: @offset = global i32 8, align 4
