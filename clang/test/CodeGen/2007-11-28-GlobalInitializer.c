// RUN: %clang_cc1 -emit-llvm %s -o - 
// PR1744
typedef struct foo { int x; char *p; } FOO;
extern FOO yy[];

int *y = &((yy + 1)->x);
void *z = &((yy + 1)->x);

