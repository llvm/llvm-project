// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

void query_newnamebuf(void) { ((void)"query_newnamebuf"); }

