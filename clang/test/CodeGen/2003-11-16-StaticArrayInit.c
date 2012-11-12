// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

void bar () {
 static char x[10];
 static char *xend = x + 10;
}


