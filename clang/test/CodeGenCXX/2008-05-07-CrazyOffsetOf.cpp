// RUN: %clang_cc1 -emit-llvm %s -o -
// rdar://5914926

struct bork {
  struct bork *next_local;
  char * query;
};
int offset =  (char *) &(((struct bork *) 0x10)->query) - (char *) 0x10;
